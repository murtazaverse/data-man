from langchain.tools import tool

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

    # Clean the response to extract only the code part
    code = response.split("```")[1] if "```" in response else response
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
