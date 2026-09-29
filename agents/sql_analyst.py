import os

import psycopg2
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from models.schema import AgentSchema, JudgeSchema
from utils.database import DatabaseUtils
from utils.jev_decision import classify_request, score_table_relevance
from utils.llm_selection import pick_llm

load_dotenv()


def database_config() -> dict[str, str]:
    """Read the existing PostgreSQL connection settings."""
    return {
        "host": os.environ["host"],
        "port": os.environ["port"],
        "user": os.environ["user"],
        "password": os.environ["password"],
        "dbname": os.environ["database"],
    }


def load_table_catalog(state: AgentSchema) -> dict:
    """Read table and column names so JEV knows the connected database's scope."""
    database = DatabaseUtils(database_config())
    try:
        catalog = database.table_catalog(state.database_schema)
    except (ConnectionError, psycopg2.Error):
        answer = "I could not read the database tables. Please check the connection and schema."
        return {"final_answer": answer, "messages": [AIMessage(content=answer)]}
    finally:
        database.close()

    if not catalog:
        answer = f"I found no tables in the {state.database_schema!r} database schema."
        return {"final_answer": answer, "messages": [AIMessage(content=answer)]}
    return {"table_catalog": catalog}


def classify_request_jev(state: AgentSchema) -> dict:
    """Use JEV Choice to route the request and Score to rate its complexity."""
    if not state.user_question.strip():
        return {"request_route": "clarification"}
    decision = classify_request(state.user_question, state.table_catalog)
    return {
        "request_route": decision.route,
        "route_confidence": decision.confidence,
        "complexity_score": decision.complexity_score,
    }


def clarification(state: AgentSchema) -> dict:
    """Ask for the missing detail before generating SQL."""
    answer = "Could you clarify what you want to know about the available data?"
    return {"final_answer": answer, "messages": [AIMessage(content=answer)]}


def explain_scope(state: AgentSchema) -> dict:
    """Explain scope using the connected database, without fixed domain examples."""
    tables = ", ".join(table["name"] for table in state.table_catalog)
    answer = f"I can answer questions about data in these tables: {tables}."
    return {"final_answer": answer, "messages": [AIMessage(content=answer)]}


def score_relevant_tables(state: AgentSchema) -> dict:
    """Use JEV Noul to suggest useful tables while allowing joins to others."""
    scores = score_table_relevance(state.curated_ques, state.table_catalog)
    ranked = sorted(scores, key=scores.get, reverse=True)
    top_names = set(ranked[: min(3, len(ranked))])
    selected = [
        table["name"]
        for table in state.table_catalog
        if table["name"] in top_names or scores[table["name"]] >= 0.55
    ]
    return {"relevance_scores": scores, "selected_tables": selected}


def curate_ques(state: AgentSchema) -> dict:
    """Rewrite the user's question for the SQL generation step."""
    model = pick_llm("low")
    response = model.invoke(f"Curate the following question: {state.user_question}").content
    return {"curated_ques": response}


def prompt_query_context(state: AgentSchema) -> dict:
    """Build an SQL prompt with JEV hints and the current database schema."""
    curated_ques = state.curated_ques

    obj = DatabaseUtils(database_config())
    try:
        schema_details = obj.schema_details(state.database_schema)
    finally:
        obj.close()

    complexity_hint = (
        "This question may need multiple joins or analysis steps; plan those carefully."
        if state.complexity_score >= 1.5
        else "Start with the simplest query that answers the question."
    )
    table_hint = ", ".join(state.selected_tables)

    prompt = f"""
    You are an SQL analyst agent. Your task is to convert the user's natural language
    query into Postgres SQL query that can be executed on the database. You are provided
    with the user's original query and the schema details of the database, including
    table names, column names, data types, and sample data for each table so that
    you can understand the structure of the database and generate an accurate SQL query.
    Unless user explicitly asks for specific number of rows, always limit the output to 10 rows.
    Note - Just generate the SQL query without any explanation or additional text because
    this query will be executed directly on the database. So, the output should be SQL
    ready to be executed without any modifications.

    User's Original Query: {curated_ques}

    JEV-suggested tables: {table_hint}
    Prefer these tables, but use other schema tables if needed for joins or accuracy.
    {complexity_hint}

    Database Schema Details:
    {schema_details}
    """

    return {"prompt_query_context": prompt}


def generate_sql(state: AgentSchema) -> dict:
    """Generate SQL from the question and database context."""

    prompt = state.prompt_query_context

    llm = pick_llm("medium")
    generated_sql_query = llm.invoke(prompt).content

    return {"generated_sql_query": generated_sql_query}


def is_safe_sql(state: AgentSchema) -> dict:
    """Ask the existing LLM judge whether to allow the generated SQL."""

    sql_query = state.generated_sql_query

    llm = pick_llm("medium")
    llm_judge = llm.with_structured_output(JudgeSchema, method="json_schema")

    prompt = f"""
    You are an SQL Judge for data security. Your task is to determine whether the SQL query is
    safe or not. The SQL query should only be used for data retrieval and should not modify the
    database in any way. Neither the SQL query nor the prompt should contain any SQL commands that can modify the
    database, such as INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, or any other commands that can change
    the structure or content of the database. If the SQL query is safe, respond with 'Yes' otherwise respond with
    'No'. Additionally, provide comments explaining your decision.
    Here's the SQL query to evaluate:
    {sql_query}"""

    response = llm_judge.invoke(prompt).model_dump()
    return {"is_safe": response["answer"], "comments": response["comments"]}


def canceled_sql(state: AgentSchema) -> dict:
    """Explain why the judge rejected the SQL."""
    comments = state.comments

    answer = f"The generated SQL query was deemed unsafe to execute. The reason provided by the judge is: {comments}. Therefore, the SQL query will not be executed."
    return {"final_answer": answer, "messages": [AIMessage(content=answer)]}


def execute_sql(state: AgentSchema) -> dict:
    """Run SQL only after the judge allows it."""
    sql_query = state.generated_sql_query

    obj = DatabaseUtils(database_config())

    execution_result = obj.execute_sql(sql_query)

    return {"sql_query_execution_result": execution_result}


def represent_final_answer(state: AgentSchema) -> dict:
    """Turn the database result into a user-facing answer."""
    execution_result = state.sql_query_execution_result
    curated_question = state.curated_ques

    llm = pick_llm("low")

    prompt = f"""
    You are an SQL analyst agent. Your task is to provide a final answer to the user based on the
    execution result of the SQL query and the user's original question. The final answer should be
    concise, clear, and directly address the user's query. Avoid including any SQL code or technical
    details in the final answer. The final answer should be in a user-friendly format that is easy to
    understand. If the execution result is empty or does not provide a clear answer to the user's question, explain this in the final answer. \n
    Here is the execution result: {execution_result} \n
    Here is the user's original question: {curated_question}
    """

    llm_response = llm.invoke(prompt).content

    return {"final_answer": llm_response, "messages": [AIMessage(content=llm_response)]}


sql_agent_graph = StateGraph(AgentSchema)
sql_agent_graph.add_node("load_table_catalog", load_table_catalog)
sql_agent_graph.add_node("classify_request_jev", classify_request_jev)
sql_agent_graph.add_node("clarification", clarification)
sql_agent_graph.add_node("explain_scope", explain_scope)
sql_agent_graph.add_node("curate_ques", curate_ques)
sql_agent_graph.add_node("score_relevant_tables", score_relevant_tables)
sql_agent_graph.add_node("prompt_query_context", prompt_query_context)
sql_agent_graph.add_node("generate_sql", generate_sql)
sql_agent_graph.add_node("is_safe_sql", is_safe_sql)
sql_agent_graph.add_node("canceled_sql", canceled_sql)
sql_agent_graph.add_node("execute_sql", execute_sql)
sql_agent_graph.add_node("represent_final_answer", represent_final_answer)

sql_agent_graph.add_edge(START, "load_table_catalog")
sql_agent_graph.add_conditional_edges(
    "load_table_catalog",
    lambda state: END if state.final_answer else "classify_request_jev",
    {END: END, "classify_request_jev": "classify_request_jev"},
)
sql_agent_graph.add_conditional_edges(
    "classify_request_jev",
    lambda state: state.request_route,
    {
        "sql_question": "curate_ques",
        "clarification": "clarification",
        "out_of_scope": "explain_scope",
    },
)
sql_agent_graph.add_edge("clarification", END)
sql_agent_graph.add_edge("explain_scope", END)
sql_agent_graph.add_edge("curate_ques", "score_relevant_tables")
sql_agent_graph.add_edge("score_relevant_tables", "prompt_query_context")
sql_agent_graph.add_edge("prompt_query_context", "generate_sql")
sql_agent_graph.add_edge("generate_sql", "is_safe_sql")
sql_agent_graph.add_conditional_edges(
    "is_safe_sql",
    lambda state: "execute_sql" if state.is_safe == "Yes" else "canceled_sql",
    {
        "execute_sql": "execute_sql",
        "canceled_sql": "canceled_sql",
    },
)
sql_agent_graph.add_edge("canceled_sql", END)
sql_agent_graph.add_edge("execute_sql", "represent_final_answer")
sql_agent_graph.add_edge("represent_final_answer", END)


sql_analyst = sql_agent_graph.compile()


if __name__ == "__main__":
    input_schema = {"user_question": "what is the most common driver name?"}
    response = sql_analyst.invoke(input_schema)
    print(response["final_answer"], response["generated_sql_query"])
