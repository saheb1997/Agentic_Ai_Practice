from dotenv import load_dotenv
import os, traceback
from langchain_openai import AzureChatOpenAI 
from typing import TypedDict,Annotated,Literal,Optional
from langgraph.graph import StateGraph,START,END
import operator
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser,StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableLambda,RunnableBranch

load_dotenv()  # ensure .env loaded

# Instantiate explicitly with api_key to avoid env name issues

model = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        temperature=0.2,
        api_key=os.getenv("AZURE_OPENAI_KEY"),  # explicit
    )


promt1 = PromptTemplate(
    template= 'write a detailed report on {topic}',
    input_variables= ['topic']
)


promt2 = PromptTemplate(
    template = 'sumarize the following text \n{text}',
    input_variables= ['text']
)

parser =StrOutputParser()


report_generation_chain = promt1 | model | parser 

branch_chain = RunnableBranch(
    (lambda x : len(x.split())>500 , promt2 |model |parser),
    RunnablePassthrough()
)

final_chain = report_generation_chain | branch_chain

result = final_chain.invoke({'topic':'crusing bikes of India'})

print(result)