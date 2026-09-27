import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from models.schema import AgentSchema, JudgeSchema
from utils.database import DatabaseUtils
from utils.llm_selection import pick_llm

load_dotenv()


# --------------- AI Code ---------------


def curate_ques(state: AgentSchema) -> AgentSchema:
    """
    *FIRST NODE*

    This just curates the user's question.
    Args: Taking in the State i.e. Agent Schema.
    Return: Also returns the state.

    Uses Low level model.
    """
    model = pick_llm("low")
    response = model.invoke(f"Curate the following question: {state.user_question}").content
    state.curated_ques = response

    # Also setting the messages here
    state.messages = state.messages + [
        HumanMessage(content={response})
    ]  # Appended the curated question to the message list

    return state


def prompt_query_context(state: AgentSchema) -> AgentSchema:
    """
    *SECOND NODE*
    Here, we are getting the curated state from the previous node,
    And adding the schema with the curated question.
    """

    # Starting from the output of previous node
    curated_ques = state.curated_ques

    # setup the connection as we are gonna be making a db call
    conn = {
        "host": os.environ["host"],
        "port": os.environ["port"],
        "user": os.environ["user"],
        "password": os.environ["password"],
        "dbname": os.environ["database"],
    }

    obj = DatabaseUtils(conn)

    schema_details = obj.schema_details("public")  # Fetch schema details from the 'public' schema

    # Constructing the prompt query for the agent to generate the SQL query
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

    Database Schema Details:
    {schema_details}
    """

    # Setting up the prompt query context for LLM
    state.prompt_query_context = prompt

    return state


# Generate SQL Query Node
def generate_sql(state: AgentSchema) -> AgentSchema:
    """
    *THIRD NODE*

    This node generates the SQL query for our LLM.
    """

    prompt = state.prompt_query_context

    # Now, the prompt is ready, send it to LLM.
    llm = pick_llm("medium")
    generated_sql_query = llm.invoke(prompt).content

    # Updating the state
    state.generated_sql_query = generated_sql_query

    return state


# Now, we wanna check if this query is safe to execute or not (AI Security comes into play)
# LLM as a Judge.


# Is safe Node
def is_safe_sql(state: AgentSchema) -> AgentSchema:
    # Can implement JEV here.
    """
    *FOURTH NODE*

    This Node keeps an LLM-as-a-Judge.
    This takes the decision whether a query is safe to execute or not.
    """

    sql_query = state.generated_sql_query

    llm = pick_llm("medium")
    llm_judge = llm.with_structured_output(JudgeSchema, method="json_schema")

    # Setting up prompt for LLM to judge
    prompt = f"""
    You are an SQL Judge for data security. Your task is to determine whether the SQL query is
    safe or not. The SQL query should only be used for data retrieval and should not modify the
    database in any way. Neither the SQL query nor the prompt should contain any SQL commands that can modify the
    database, such as INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, or any other commands that can change
    the structure or content of the database. If the SQL query is safe, respond with 'Yes' otherwise respond with
    'No'. Additionally, provide comments explaining your decision.
    Here's the SQL query to evaluate:
    {sql_query}"""

    response = llm_judge.invoke(prompt).model_dump()  # Get the structured output as a dictionary
    state.is_safe = response["answer"]  # yes/no response.
    state.comments = response["comments"]  # reasoning

    return state


# Now, we wanna make a conditonal node i.e. "RUN"
# This node will only run this particular query if the answer is YES.
# Otherwise, will return END the agent, and reason why it got ended.


# Canceled SQL Query Node
def canceled_sql(state: AgentSchema) -> AgentSchema:

    comments = state.comments

    state.final_answer = f"The generated SQL query was deemed unsafe to execute. The reason provided by the judge is: {comments}. Therefore, the SQL query will not be executed."
    state.messages = state.messages + [
        AIMessage(content=f"{state.final_answer}")
    ]  # Append the final answer to the messages list, AI Message because this is an AI message
    # Q: Do we need to pass the chat history aswell, whenever we are talking to this particular agent?
    # A: Ideally yes, as we can see that this agent is performing all of the things above, and have all the context, it can actually better answer you everything. But the thing is , this is not an independent agent, this is a subagent, so ideally the chat should be maintained at the parent level (which I'd be adding later). So, we do not need to send the chat directly from the Agent, but from outside the agent, so that  it can inherit that entire chat history. Because, what will happen, everytime this agent will be called, this will be getting the parameters from the outside, but when we are just executing the agent, just for the sake of testing it, we can give all the information on our own. In this agent, we wont be passing the chat history, we'll be passing from the outside.
    return state


# Execute SQL Query Node
def execute_sql(state: AgentSchema) -> AgentSchema:

    sql_query = state.generated_sql_query

    conn_details = {
        "host": os.environ["host"],
        "port": os.environ["port"],
        "user": os.environ["user"],
        "password": os.environ["password"],
        "dbname": os.environ["database"],
    }

    obj = DatabaseUtils(conn_details)

    execution_result = obj.execute_sql(sql_query)  # Execute the SQL query on the database

    state.sql_query_execution_result = execution_result

    return state


# Represent the final answer Node
def represent_final_answer(state: AgentSchema) -> AgentSchema:

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

    llm_response = llm.invoke(prompt).content  # Get the final answer from the LLM

    state.final_answer = llm_response
    state.messages = state.messages + [
        AIMessage(content=f"{llm_response}")
    ]  # Append the final answer to the messages list

    return state


# Q: Do we need to pass the chat history aswell, whenever we are talking to this particular agent?
# A: Ideally yes, as we can see that this agent is performing all of the things above, and have all the context, it can actually better answer you everything. But the thing is , this is not an independent agent, this is a subagent, so ideally the chat should be maintained at the parent level (which I'd be adding later). So, we do not need to send the chat directly from the Agent, but from outside the agent, so that  it can inherit that entire chat history. Because, what will happen, everytime this agent will be called, this will be getting the parameters from the outside, but when we are just executing the agent, just for the sake of testing it, we can give all the information on our own.


# =================== GRAPH BUILDER ==============

sql_agent_graph = StateGraph(AgentSchema)  # The thing whose state we need to track.

# Now its time to make the nodes, prompt_query_contexti.e. the functions we have defined above.
sql_agent_graph.add_node(curate_ques, "curate_ques")
sql_agent_graph.add_node(prompt_query_context, "prompt_query_context")
sql_agent_graph.add_node(generate_sql, "generate_sql")
sql_agent_graph.add_node(is_safe_sql, "is_safe_sql")
sql_agent_graph.add_node(canceled_sql, "canceled_sql")
sql_agent_graph.add_node(execute_sql, "execute_sql")
sql_agent_graph.add_node(represent_final_answer, "represent_final_answer")

# Now the edges, i.e. the flow of the graph.
sql_agent_graph.add_edge(START, "curate_ques")
sql_agent_graph.add_edge("curate_ques", "prompt_query_context")
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


# Compile the graph to ensure all nodes and edges are valid
sql_analyst = sql_agent_graph.compile()


if __name__ == "__main__":
    # Optional
    from IPython.display import Image

    img = Image(sql_analyst.get_graph().draw_mermaid_png())
    with open("sql_analyst_graph.png", "wb") as f:
        f.write(img.data)

    # Now to test the agent.
    # Setting up Input Schema
    input_schema = {
        "messages": [],
        "user_question": "What is the average rating given to drivers?",  # Question in layman terms, which will be converted to SQL query by the agent.
        "curated_ques": "",
        "prompt_query_context": "",
        "is_safe": "No",
        "generated_sql_query": "",
        "sql_query_execution_result": "",
        "final_answer": "",
        "comments": "",
    }
    # Note: The rest of the values are empty as they will be filled in by the agent as it processes the input.
    response = sql_analyst.invoke(input_schema)  # Invoke the agent with the input schema.
    print(response["final_answer"])  # Print only the final response from the agent.
