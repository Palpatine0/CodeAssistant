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

chain.invoke({"question": "How to create a RAG chain in LCEL?"})