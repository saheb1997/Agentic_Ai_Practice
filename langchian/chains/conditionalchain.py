from dotenv import load_dotenv
import os, traceback
from langchain_openai import AzureChatOpenAI 
from typing import TypedDict,Annotated,Literal,Optional
from langgraph.graph import StateGraph,START,END
import operator
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser,StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableBranch,RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal 


load_dotenv()  # ensure .env loaded

# Instantiate explicitly with api_key to avoid env name issues

model = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        temperature=0.2,
        api_key=os.getenv("AZURE_OPENAI_KEY"),  # explicit
    )
class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Give the sentiment of the feedback"
    )

parser = StrOutputParser()
parser2 = PydanticOutputParser(pydantic_object = Feedback)

promt1 = PromptTemplate(
    template='clasify the sentiment of the following text to positive and negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions() }

)




classifier_chain = promt1 | model | parser2
 
# result = classifier_chain.invoke({'feedback':'This is a bad phone'}).sentiment

# print(result)

promt2 =PromptTemplate(
    template='Write an appropiate response to this positive feedback  \n {feedback}',
    input_variables=['feedback']
)

promt3 =PromptTemplate(
    template='Write an appropiate response to this negetive feedback  \n {feedback}',
    input_variables=['feedback']
)
branch_chain = RunnableBranch(
    (lambda x: x.sentiment=='positive', promt2| model | parser),
    (lambda x: x.sentiment=='negative', promt3| model | parser),
    RunnableLambda( lambda x: "could not find sentiment")
)

chain= classifier_chain | branch_chain

print(chain.invoke({'feedback':'This is a very bad phone'}))

