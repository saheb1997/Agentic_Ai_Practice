from dotenv import load_dotenv
import os, traceback
from langchain_openai import AzureChatOpenAI 
from typing import TypedDict,Annotated,Literal,Optional
from langgraph.graph import StateGraph,START,END
import operator
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser,StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableLambda,RunnableBranch
from langchain_community.document_loaders import TextLoader,DirectoryLoader, PyPDFLoader


load_dotenv()  # ensure .env loaded

# Instantiate explicitly with api_key to avoid env name issues

model = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        temperature=0.2,
        api_key=os.getenv("AZURE_OPENAI_KEY"),  # explicit
    )

loader = DirectoryLoader(
    path='pdfs',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)


docs = loader.lazy_load()

for document in docs:
    print(document.metadata)
