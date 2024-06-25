from bs4 import BeautifulSoup as Soup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader

from dotenv import load_dotenv
from langchain.callbacks.tracers.langchain import wait_for_all_tracers

load_dotenv()
wait_for_all_tracers()

# LCEL docs
url = "https://python.langchain.com/v0.2/docs/concepts/#langchain-expression-language-lcel"
loader = RecursiveUrlLoader(
    url = url, max_depth = 20, extractor = lambda x: Soup(x, "html.parser").text
)
docs = loader.load()

# Sort the list based on the URLs and get the text
d_sorted = sorted(docs, key = lambda x: x.metadata["source"])
d_reversed = list(reversed(d_sorted))
concatenated_content = "\n\n\n --- \n\n\n".join(
    [doc.page_content for doc in d_reversed]
)


from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.utils.function_calling import convert_to_openai_tool

## Data model
class code(BaseModel):
    """Code output"""
    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")

## LLM
model = ChatOpenAI(temperature=0, streaming=True)

## Tool
code_tool_oai = convert_to_openai_tool(code)

## LLM with tool and enforce invocation
llm_with_tool = model.bind(
    tools=[code_tool_oai],
    tool_choice={"type": "function", "function": {"name": "code"}},
)

# Parser
parser_tool = PydanticToolsParser(tools=[code])

## Prompt
template = """
You are a coding assistant with expertise in LCEL, LangChain expression language. 
Here is a full set of LCEL documentation: 
\n _______ \n 
{context} 
\n _______ \n 
Answer the user question based on the above provided documentation. 
Ensure any code you provide can be executed with all required imports and variables defined. 
Structure your answer with a description of the code solution. 
Then list the imports. And finally list the functioning code block. 
Here is the user question: \n _______ \n {question}
"""

## Prompt
prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"],
)

# Chain
chain = (
    {
        "context": lambda x: docs,
        "context": lambda x: concatenated_content,
        "question": itemgetter("question"),
    }
    | prompt
    | llm_with_tool
    | parser_tool
)


from typing import Dict, TypedDict
from langchain_core.messages import BaseMessage

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        keys: A dictionary where each key is a string.
    """
    keys: Dict[str, any]


from operator import itemgetter
from bs4 import BeautifulSoup as Soup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.utils.function_calling import convert_to_openai_tool


def generate(state):
    """
    Generate a code solution based on LCEL docs and the input question
    with optional feedback from code execution tests

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """

    ## State
    state_dict = state["keys"]
    question = state_dict["question"]
    iter = state_dict["iterations"]

    ## Data model
    class code(BaseModel):
        """"
        Code output
        """
        prefix: str = Field(description = "Description of the problem and approach")
        imports: str = Field(description = "Code block import statements")
        code: str = Field(description = "Code block not including import statements")

    ## LLM
    model = ChatOpenAI(temperature = 0, streaming = True)

    # Tool
    code_tool_oai = convert_to_openai_tool(code)

    # LLM with tool and enforce invocation
    llm_with_tool = model.bind(
        tools = [convert_to_openai_tool(code_tool_oai)],
        tool_choice = {"type": "function", "function": {"name": "code"}},
    )

    # Parser
    parser_tool = PydanticToolsParser(tools = [code])

    ## Prompt
    template = """"
    You are a coding assistant with expertise in LCEL, LangChain expression language. 
    Here is a full set of LCEL documentation:

    --------
    {context}
    --------
    Answer the user question based on the above provided documentation. 
    Ensure any code you provide can be executed with all required imports and variables defined. 
    Structure your answer with a description of the code solution.
    Then list the imports. And finally list the functioning code block.
    Here is the user question: 
    ----
    {question}
    """
    # if "error" in state_dict:
    if "error" in state_dict:
        print("--RE-GENERATE SOLUTION w/ ERROR FEEDBACK--")
        error = state_dict["error"]
        code_solution = state_dict["generation"]

        # Update prompt
        addendum = """\n ---- ---- ---- \n You previously tried to solve this problem. \n Here is your solution: \n ---- ---- ---- \n {generation} \n ---- ---- ---- \n Here is the resulting error from code execution: \n ---- ---- ---- \n {error} \n ---- ---- ---- \n Please re-try to answer this.
        Structure your answer with a description of the code solution. \n Then list the imports. And finally list the functioning code block. Structure your answer with a description of the code solution. \n Then list the imports. And finally list the functioning code block. \n Here is the user question: \n ---- ---- ---- \n {question}"""
        template = template + addendum

        # Prompt
        prompt = PromptTemplate(
            template = template,
            input_variables = ["context", "question", "generation", "error"],
        )

        # Chain
        chain = (
                {
                    "context": lambda x: concatenated_content,
                    "question": itemgetter("question"),
                    "generation": itemgetter("generation"),
                    "error": itemgetter("error"),
                }
                | prompt
                | llm_with_tool
                | parser_tool
        )

        code_solution = chain.invoke({"question": question, "generation": str(code_solution[0]), "error": error})
    else:
        print("---GENERATE SOLUTION---")

        # Prompt
        prompt = PromptTemplate(
            template = template,
            input_variables = ["context", "question"],
        )

        # Chain
        chain = (
                {
                    "context": lambda x: concatenated_content,
                    "question": itemgetter("question"),
                }
                | prompt
                | llm_with_tool
                | parser_tool
        )

        code_solution = chain.invoke({"question": question})

    iter = iter + 1
    return {"keys": {"generation": code_solution, "question": question, "iterations": iter}}


def check_code_imports(state):
    """
    Check imports

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, error
    """

    ## State
    print("--CHECKING CODE IMPORTS--")
    state_dict = state["keys"]
    question = state_dict["question"]
    code_solution = state_dict["generation"]
    imports = code_solution[0].imports
    iter = state_dict["iterations"]

    try:
        # Attempt to execute the imports
        exec(imports)
        # No errors occurred
        error = "None"
    except Exception as e:
        print("--CODE IMPORT CHECK: FAILED--")
        # Catch any error during execution (e.g., ImportError, SyntaxError)
        error = f"Execution error: {e}"
        if "error" in state_dict:
            error_prev_runs = state_dict["error"]
            error = error_prev_runs + "\n -- Most recent run error -- \n" + error

    return {"keys": {"generation": code_solution, "question": question, "error": error, "iterations": iter}}




def check_code_execution(state):
    """
    Check code block execution

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, error
    """

    ## State
    print("---CHECKING CODE EXECUTION---")
    state_dict = state["keys"]
    question = state_dict["question"]
    code_solution = state_dict["generation"]
    prefix = code_solution[0].prefix
    imports = code_solution[0].imports
    code = code_solution[0].code
    code_block = imports + "\n" + code
    iter = state_dict["iterations"]

    try:
        # Attempt to execute the code block
        exec(code_block)
    except Exception as e:
        print("---CODE BLOCK CHECK: FAILED---")
        # Catch any error during execution (e.g., ImportError, SyntaxError)
        error = f"Execution error: {e}"
        if "error" in state_dict:
            error_prev_runs = state_dict["error"]
            error = error_prev_runs + "\n --- Most recent run error --- \n" + error
    else:
        print("---CODE BLOCK CHECK: SUCCESS---")
        # No errors occurred
        error = "None"

    return {"keys": {"generation": code_solution, "question": question, "error": error, "prefix": prefix,
                     "imports": imports, "iterations": iter, "code": code}}


def decide_to_check_code_exec(state):
    """
    Determines whether to test code execution, or re-try answer generation.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    print("---DECIDE TO TEST CODE EXECUTION---")
    state_dict = state["keys"]
    question = state_dict["question"]
    code_solution = state_dict["generation"]
    error = state_dict["error"]

    if error == "None":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: TEST CODE EXECUTION---")
        return "check_code_execution"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: RE-TRY SOLUTION---")
        return "generate"


def decide_to_finish(state):
    """
    Determines whether to finish (re-try code 3 times).

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    print("---DECIDE TO TEST CODE EXECUTION---")
    state_dict = state["keys"]
    question = state_dict["question"]
    code_solution = state_dict["generation"]
    error = state_dict["error"]
    iter = state_dict["iterations"]

    if error == "None" or iter == 3:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: TEST CODE EXECUTION---")
        return "end"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: RE-TRY SOLUTION---")
        return "generate"

from langgraph.graph import END, StateGraph

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("generate", generate)  # generation solution
workflow.add_node("check_code_imports", check_code_imports)  # check imports
workflow.add_node("check_code_execution", check_code_execution)  # check execution

# Build graph
workflow.set_entry_point("generate")
workflow.add_edge("generate", "check_code_imports")
workflow.add_conditional_edges(
    "check_code_imports",
    decide_to_check_code_exec,
    {
        "check_code_execution": "check_code_execution",
        "generate": "generate",
    },
)
workflow.add_conditional_edges(
    "check_code_execution",
    decide_to_finish,
    {
        "end": END,
        "generate": "generate",
    },
)

# Compile
app = workflow.compile()

question = "How can I directly pass a string to a runnable and use it to construct the input needed for my prompt?"

config = {"recursion_limit": 50}
answer = app.invoke({"keys": {"question": question, "iterations": 0}}, config=config)


 