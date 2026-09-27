import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from models.schema import AgentSchema, JudgeSchema
from utils.database import DatabaseUtils
from utils.llm_selection import pick_llm

load_dotenv()

# --------------- AI Code -----------


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
    state.curated_ques = response.content[0]["text"]

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

    schema_details = obj.schema_details  # Fetch schema details from the 'public' schema

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
    llm_judge = llm.with_structured_output(JudgeSchema)

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
