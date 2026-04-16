from dotenv import load_dotenv
import os, traceback
from langchain_openai import AzureChatOpenAI 
from typing import TypedDict,Annotated,Literal,Optional
from langgraph.graph import StateGraph,START,END
import operator
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser,StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableLambda

load_dotenv()  # ensure .env loaded

# Instantiate explicitly with api_key to avoid env name issues

model = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        temperature=0.2,
        api_key=os.getenv("AZURE_OPENAI_KEY"),  # explicit
    )

def wordcount(text):
    return len(text.split())

prompt= PromptTemplate(
    template = 'Write a joke about {topic}',
    input_variables= ['topic']
)

parser = StrOutputParser()

joke_gen_chain = prompt |model |parser

parallel_chain = RunnableParallel(
    {
        'joke': RunnablePassthrough(),
        'word_count': RunnableLambda(wordcount)
    }
)

final_chain = joke_gen_chain | parallel_chain

result = final_chain.invoke({'topic':'AI'})
print(result['joke'])

print(result['word_count'])