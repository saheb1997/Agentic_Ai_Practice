from dotenv import load_dotenv
import os, traceback
from langchain_openai import AzureChatOpenAI 
from typing import TypedDict,Annotated,Literal,Optional
from langgraph.graph import StateGraph,START,END
import operator
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser,StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

load_dotenv()  # ensure .env loaded

# Instantiate explicitly with api_key to avoid env name issues

model = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        temperature=0.2,
        api_key=os.getenv("AZURE_OPENAI_KEY"),  # explicit
    )


prompt1= PromptTemplate(
    template='Generate a tweet about {topic}',
    input_variable = ['topic']
)

prompt2 = PromptTemplate(
     template= 'Generate a Linkedin post about {topic}',
     input_variables =['topic']
)


parser = StrOutputParser()
joke_gen_chain = prompt1 | model | parser


passthrough  = RunnablePassthrough ()

parallel_chain = RunnableParallel(
    {
        'joke':RunnablePassthrough(),
        'explanation':prompt2 | model | parser
    }
)

final_chain = joke_gen_chain | parallel_chain 


result =final_chain.invoke({'topic':"cricket"})
print(result['joke'])
print("\n\n")
print(result['explanation'])
