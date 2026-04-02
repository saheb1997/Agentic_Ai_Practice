from dotenv import load_dotenv
import os, traceback
from langchain_openai import AzureChatOpenAI 
from typing import TypedDict,Annotated,Literal,Optional
from langgraph.graph import StateGraph,START,END
import operator
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser,StrOutputParser

load_dotenv()  # ensure .env loaded

# Instantiate explicitly with api_key to avoid env name issues

model = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        temperature=0.2,
        api_key=os.getenv("AZURE_OPENAI_KEY"),  # explicit
    )

prompt1 = PromptTemplate(
    template='make a detail report for this: {topic}',
    input_variables=['topic']
)
promt2 = PromptTemplate(
    template='Give 5 important points from generated report {text}',
    input_variables=['text']
)

parser =StrOutputParser()


chain= prompt1 | model | parser | promt2 | model |parser


result = chain.invoke({'topic':'Ai jobs'})

print(result)