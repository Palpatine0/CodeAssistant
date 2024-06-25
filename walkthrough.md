## Code Assistant

### Project Introduction

<div align="center">
<img height="300" src="https://i.imghippo.com/files/XsEXc1719331190.jpg" alt="" border="0">
</div>


In this project, we explore the development of a self-corrective coding assistant using Lang Graph and large language
models (LLMs). The focus is on implementing concepts from the [Alpha Codium paper](https://arxiv.org/pdf/2401.08500),
which introduces flow engineering for iterative code generation and testing.

By leveraging Lang Graph, we aim to create a system that generates code solutions, tests them, and iteratively improves
the solutions based on test results. This approach moves beyond naive prompt responses to a more sophisticated flow
where solutions are refined through structured feedback and reflection.

The project demonstrates the practical application of these concepts and evaluates the performance improvements achieved
through this methodology.

### Prerequisites

- Python 3.11
- pip (Python package installer)
- Git (optional)

### Step 1: Initial Setup

#### 1. Initialize the Environment

First, let's set up the environment and install necessary dependencies.

1. **Create a `.env` file:**

2. This file will store your API keys and other configuration settings. Ensure it is included in your `.gitignore` file
   to prevent it from being committed to your repository.

   Example `.env` file:
   ```plaintext
   LANGCHAIN_API_KEY="your_langchain_api_key"
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
   LANGCHAIN_PROJECT="CodeAssistant"
   
   OPENAI_API_KEY="your_openai_api_key"
   ```

2. **Install required packages:**
   ```bash
   pip install langchain langchain_community langchain_openai openai python-dotenv  
   ```
   ```bash
   pip install bs4
   ```
   ```bash
   pip install -U langchain-cli
   ```
   ```bash
   pip install langchain-anthropic
   ```
   ```bash
   pip install langgraph
   ```

#### Key Concepts

##### 1. Anthropic

- **Definition**: Anthropic is a company specializing in creating advanced AI models and tools, focusing on developing
  responsible AI systems that align with human values and ethical standards. They are known for their work on language
  models and AI safety research.
- **Usage**: Anthropic's AI models and tools are integrated into applications to provide robust and ethical AI
  solutions. These models are used for various natural language processing tasks, ensuring that AI systems are not only
  effective but also align with ethical guidelines and human-centered values.

### Step 2: Setup LangServe and LangSmith

#### 1. LangServe Setup

Set up LangServe to manage our application deployment.
Use the LangServe CLI to create a new application called `code-assistant`.

```bash
langchain app new code-assistant
```   

#### 2. LangSmith Setup

Make sure u have created a LangSmith project for this lab.

**Project Name:** CodeAssistant

### Step 3: Add RecursiveUrlLoader for LCEL Docs Retrieval and Processing

<div align="center">
<img height="300" src="https://i.imghippo.com/files/wtW3F1719331441.jpg" alt="" border="0">
</div>

In this step, we will integrate the `RecursiveUrlLoader` to load documents from the specified URL. We will then process
these documents by sorting, reversing, and concatenating their content for better readability.

#### 1. Create `chain.py` to Integrate RecursiveUrlLoader and Process Documents

Here, we set up the necessary components to load, sort, and concatenate documents from a specified URL.

**File**: `code-assistant/app/chain.py`

**Code for `chain.py`**:

```python
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

print(concatenated_content)
```

In this `chain.py` file:

- **RecursiveUrlLoader** is used to retrieve content from a specified URL recursively up to a certain depth.
- **BeautifulSoup** is used to extract and parse the HTML content.
- The documents are sorted based on their URLs and then reversed.
- The content of the sorted documents is concatenated with separators for better readability.

#### 2. Test the Chain

Run the `chain.py` file and inspect the results to ensure that the documents are retrieved, processed, and concatenated
correctly.

<img src="https://i.imghippo.com/files/WZt8j1719329591.jpg" alt="" border="0">

#### Key Concepts

##### 1. RecursiveUrlLoader

- **Definition**: `RecursiveUrlLoader` is a tool that loads documents from a specified URL, navigating through links
  recursively up to a given depth.
- **Usage**: It is used to retrieve and process content from websites, allowing for in-depth data collection from
  multiple linked pages. This is particularly useful for extracting comprehensive information from interconnected
  documents or web pages.

### Step 4: Implement LLM Prompt Chain with LCEL Documentation and User Question Handling

<div align="center">
<img height="300" src="https://www.imghippo.com/i/fOYDZ1719331510.jpg" alt="" border="0">
</div>


In this step, we will implement a new LLM prompt chain that utilizes the LangChain expression language (LCEL)
documentation to handle user questions. This involves setting up a chain that processes the documentation, formats the
user question, and generates a structured code solution using an LLM.

**File**: `code-assistant/app/chain.py`

**Updated Code for `chain.py`**:

```python
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

print(concatenated_content)

from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.utils.function_calling import convert_to_openai_tool


## Data model
class code(BaseModel):
    """Code output"""
    prefix: str = Field(description = "Description of the problem and approach")
    imports: str = Field(description = "Code block import statements")
    code: str = Field(description = "Code block not including import statements")


## LLM
model = ChatOpenAI(temperature = 0, streaming = True)

## Tool
code_tool_oai = convert_to_openai_tool(code)

## LLM with tool and enforce invocation
llm_with_tool = model.bind(
    tools = [code_tool_oai],
    tool_choice = {"type": "function", "function": {"name": "code"}},
)

# Parser
parser_tool = PydanticToolsParser(tools = [code])

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
    template = template,
    input_variables = ["context", "question"],
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

chain.invoke({"question": "How to create a RAG chain in LCEL?"})
```

**Explanation:**

- **Setting the Context**: The LCEL documentation is loaded, sorted, and concatenated to provide a comprehensive
  context.
- **Extracting the Question**: The user question is extracted from the input.
- **Formatting the Prompt**: The prompt template is populated with the context and question.
- **Passing to the LLM**: The formatted prompt is passed to the LLM, which uses the tool to generate a structured code
  solution.
- **Interpreting the Output**: The output from the LLM is parsed to ensure it matches the expected structure.

Overall, this setup enables the LLM to generate structured, executable code solutions based on the provided
documentation and user questions, ensuring that the outputs are well-organized and executable with all necessary imports
and code blocks.

#### 2. Test the Chain

Run the `chain.py` file and inspect the results

<img src="https://i.imghippo.com/files/hTcC71719331101.jpg" alt="" border="0">

### Step 5: Add Code Generation and Validation Workflow for LCEL

In this step, we will implement a workflow that handles code generation, import checking, and execution validation based
on LangChain expression language (LCEL). The workflow will use a state graph to manage the flow, ensuring that code
solutions are generated, tested, and refined based on error feedback.

#### 1. Create and Update `chain.py` for Code Generation and Validation

We will add functions to generate code solutions, check imports, and validate code execution. The workflow will handle
errors and re-generate solutions if necessary, iterating through the process to achieve a valid solution.

**File**: `code-assistant/app/chain.py`

**Updated Code for `chain.py`**:

```python
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
        """
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
    template = """
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


# All documents


have
been
filtered
check_relevance
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
answer = app.invoke({"keys": {"question": question, "iterations": 0}}, config = config)
```

**Explanation:**

- **Generate Function**: This function generates a code solution based on LCEL documentation and the input question. If
  there is an error from a previous attempt, it includes that feedback in the prompt to refine the solution.
- **Check Code Imports Function**: This function checks if the import statements in the generated code can be executed
  without errors. If there are errors, they are caught and returned.
- **Check Code Execution Function**: This function checks if the entire code block can be executed without errors. If
  there are errors, they are caught and returned.
- **Decision Functions**: These functions determine the next step in the workflow based on the presence of errors and
  the number of iterations.

Overall, this setup enables a comprehensive workflow that handles code generation, import checking, and execution
validation, ensuring that the generated code is correct and executable.

#### 2. Test the Chain

To test the chain, run the `chain.py` file and inspect the output to ensure that the code generation and validation
workflow is functioning correctly.

The output should include the generated code solution, any errors encountered during import and execution checks, and
any refined solutions generated based on the error feedback. This will verify that the workflow correctly handles code
generation, validation, and refinement.

<img src="https://i.imghippo.com/files/3noNw1719337661.jpg" alt="" border="0">
<img src="https://i.imghippo.com/files/3LLSM1719337310.jpg" alt="" border="0">

The following images demonstrate the workflow when a generation encounter an error and the refined solution generated

<img src="https://i.imghippo.com/files/LhsoW1719337772.png" alt="" border="0">

The first generation encountered an error, so in the next generation, this error will be passed to the second generation.

<img src="https://i.imghippo.com/files/u3Ayv1719337817.png" alt="" border="0">

The system re-generating a solution with the error feedback provided.

Overall, this setup enables the LLM to generate structured, executable code solutions based on the provided documentation and user questions, ensuring that the outputs are well-organized and executable with all necessary imports and code blocks.






