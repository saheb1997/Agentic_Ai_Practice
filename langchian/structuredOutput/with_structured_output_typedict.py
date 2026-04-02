# Diagnostic cell — run this in one cell to confirm envs and test AzureChatOpenAI
from dotenv import load_dotenv
import os, traceback
from langchain_openai import AzureChatOpenAI 
from typing import TypedDict,Annotated,Literal,Optional
from langgraph.graph import StateGraph,START,END
import operator

load_dotenv()  # ensure .env loaded

# Instantiate explicitly with api_key to avoid env name issues

model = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        temperature=0.2,
        api_key=os.getenv("AZURE_OPENAI_KEY"),  # explicit
    )




# schema
# class Review(TypedDict):
#     summary :str
#     sentiment: str


# structured_model = model.with_structured_output(Review)

# result = structured_model.invoke("The product quality is excellent and works exactly as described. The build feels durable, and the performance exceeded my expectations. Delivery was also quick and the packaging was secure. I would definitely recommend this product to others looking for something reliable.") 

# print(result)


class Review:
    key_themes: Annotated[list[str],"write down all the key themes discussed in the review in a list"]
    summary : Annotated[str,"A brief summary of the review"]
    sentiment:Annotated[Literal['pos','neg'],"Return the sentiment either possitive or negative"]
    pros:Annotated[Optional[list[str]],"Write down all the pros inside the list"]
    cons:Annotated[Optional[list[str]],"write down all the cons inside the list"]

structurd_model = model.with_structured_output(Review)
result = structurd_model.invoke("I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.ProsInsanely powerful processor (great for gaming and productivityStunning 200MP camera with incredible zoom capabilitieLong battery life with fast charginS-Pen support is unique and usefulReview by Nitish Singh")



print(result['pros'])