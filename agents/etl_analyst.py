from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain.tools import tool
from langgraph.graph import END, START, StateGraph

from models.schema import ETLAgentSchema
from utils.etl_tools import ETLTools
from utils.llm_selection import pick_llm

etl_tools = ETLTools()

# --------------------------- ETL ANALYST AGENT TOOLS ---------------------------
# This is a ReACT Agent so we do not need to maintain any special state.


@tool
def extract_load_tool(urls: list, output_folder: str, format: str) -> str:
    """Tool to extract data from a URL and loads it to a specified folder in the desired format.
    Args:
        urls (list): A list of URLs to extract data from.
        output_folder (str): The folder where the extracted data will be saved.
        format (str): The format in which to save the data. Supported formats are 'csv', 'json', and 'parquet'.
    Returns:
        str: The path to the saved file.
    """
    return etl_tools.extract_load(urls, output_folder, format)


@tool
def transform_load_tool(
    input_file_path: str, output_folder: str, output_format: str, user_question: str
) -> str:
    """Tool to fetch the context, then generate the code for transformation, and then execute the code to transform the data and save it in the desired format.
    Args:
        input_file_path (str): The path to the input file to be transformed.
        output_folder (str): The folder where the transformed data will be saved.
        output_format (str): The format in which to save the transformed data. Supported formats are 'csv', 'json', and 'parquet'.
        user_question (str): The user's question or request that describes how the data should be transformed.
    Returns:
        str: The path to the saved transformed file.
    """
    context = etl_tools.transform_load_context(input_file_path)

    # Again, this part can be done via JEV, but for now we will do it manually.
    llm = pick_llm("high")

    prompt = f"""
            You are a Python Data Analyst who uses Pandas to analyze data.
            You need to provide only the Pandas Code that will help to perform the right ETL operations on the data stored in the file : {input_file_path}
            as per the user's question. Do not provide any explanation or comments, only
            the code should be provided. The code should be in a format that can be executed
            in a Python environment with Pandas installed.
            Don't write anything else than Pandas Code. \n

            Create the Pandas Dataframe from the data stored in the file : {input_file_path} and then
            write the code to transform and save the data at {output_folder}.
            Here's the user's question: {user_question}\n
            Here's the context of the data you will be analyzing: {context}\n

        """

    response = llm.invoke(prompt)
    content = response.content if hasattr(response, "content") else response

    # Clean the response to extract only the code part
    code = content.split("```")[1] if "```" in content else content
    code = code.replace("python", "").strip()

    # Execute the code
    result = etl_tools.execute_code(code)

    return f"""The data is successfully transformed and saved at {output_folder} in {output_format} format.
                The code used for transformation is as follows:
                {code}
                The result of the code execution is as follows:
                {result}"""


# Toolkit
tools = [extract_load_tool, transform_load_tool]

# Binding tools with LLM
llm = pick_llm("high")
llm_with_tools = llm.bind_tools(tools)


# ----------------------------------- AGENT GRAPH -----------------------------


# LLM NODE
def llm_node(state: ETLAgentSchema) -> ETLAgentSchema:
    """
    The LLM Node is the brain of this network, take the user question and then will make the decision of whether to make a toolcall or not.
    """

    prompt = f"""
            You are a Python Data Analyst who has access to tools that can extract and load,
            transform and load data. You will be provided with a user's question
            and you would need to perform the right ETL operations as per the user's question.
            If the operation is performed then inform the user and end the coversation.
            Here's the chat history: {state.messages}\n
    """

    response = llm_with_tools.invoke(prompt)

    state.messages += [response]

    return state


# TOOL NODE
def tool_node(state: ETLAgentSchema) -> ETLAgentSchema:
    """
    This node is responsible for invoking the appropriate tool based on the user's question and the context provided by the LLM.
    """
    tool_results = []
    tools_by_name = {tool.name: tool for tool in tools}
    tool_calls = state.messages[-1].tool_calls

    for tool_call in tool_calls:
        tool = tools_by_name[tool_call["name"]]
        tool_result = tool.invoke(tool_call["args"])
        tool_results.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"]))

    state.messages += tool_results

    return state


# IS TOOLCALL DECISION


def is_tool_call(state: ETLAgentSchema):
    """
    This function checks if we are gonna make a toolcall or not.
    """

    tool_call_details = state.messages[-1].tool_calls

    if tool_call_details:
        return "tool_node"
    else:
        return END


# ============================== Generating the graph ==================================

# ETL Analyst Graph
etl_analyst_graph = StateGraph(ETLAgentSchema)

# Adding the nodes
etl_analyst_graph.add_node(llm_node, "llm_node")
etl_analyst_graph.add_node(tool_node, "tool_node")

# Adding the edges
etl_analyst_graph.add_edge(START, "llm_node")
etl_analyst_graph.add_conditional_edges(
    "llm_node",
    lambda state: "tool_node" if state.messages[-1].tool_calls else END,
    {
        "tool_node": "tool_node",  # value of x from lambda func
        END: END,  # value of x from lambda func
    },
)

etl_analyst_graph.add_edge("tool_node", "llm_node")


etl_analyst = etl_analyst_graph.compile()


if __name__ == "__main__":
    # prompt = f"""
    #         You are a Python Data Analyst who has access to tools that can extract and load,
    #         transform and load data. You will be provided with a user's question
    #         and you would need to perform the right ETL operations as per the user's question.
    #         If the operation is performed then inform the user and end the coversation.
    #         Here's the chat history: {[]}\n
    # """
    # state = ETLAgentSchema()
    # state.messages = []
    # response = llm_node(state)
    # print(response.messages[-1].tool_calls)

    from IPython.display import Image

    img = Image(etl_analyst.get_graph().draw_mermaid_png())
    with open("etl_analyst_graph.png", "wb") as f:
        f.write(img.data)

    response = etl_analyst.invoke(
        {
            "messages": [
                HumanMessage(
                    content="I want to extract the data from the API endpoint 'https://pokeapi.co/api/v2/pokemon' and save it to data/extract folder in the csv folder, and then transform that data into json format and save it to data/transform folder."
                )
            ]
        }
    )

    print(
        [
            message.tool_calls
            for message in response["messages"]
            if isinstance(message, AIMessage) and message.tool_calls
        ]
    )
