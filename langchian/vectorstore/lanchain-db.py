# from langchain_openai import OpenAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
import os, traceback
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI 

from langchain_core.documents import Document


load_dotenv()  # ensure .env loaded

# Instantiate explicitly with api_key to avoid env name issues



doc1 = Document(
    page_content="Adventure bikes are designed for long-distance touring and off-road capability. They usually have tall suspension, comfortable seating, and large fuel tanks.",
    metadata={"category": "ADV", "example": "Suzuki V-Strom 250"}
)

doc2 = Document(
    page_content="Cruiser bikes focus on relaxed riding posture, comfort, and low-end torque. They are ideal for highway cruising and city rides.",
    metadata={"category": "Cruiser", "example": "Royal Enfield Meteor 350"}
)

doc3 = Document(
    page_content="Bobber motorcycles have a minimalist design with chopped fenders, solo seats, and retro styling. They are customized for style and simplicity.",
    metadata={"category": "Bobber", "example": "Jawa Perak"}
)

doc4 = Document(
    page_content="Sports bikes are built for high speed, quick acceleration, and aggressive riding posture. They are commonly used for racing and performance riding.",
    metadata={"category": "Sports", "example": "Kawasaki Ninja 300"}
)

doc5 = Document(
    page_content="Touring motorcycles are made for long-distance travel with features like large storage boxes, comfortable seats, and wind protection.",
    metadata={"category": "Touring", "example": "Honda Gold Wing"}
)

doc6 = Document(
    page_content="Scrambler bikes combine retro styling with light off-road capability. They generally feature upright ergonomics and dual-purpose tires.",
    metadata={"category": "Scrambler", "example": "Triumph Scrambler 400X"}
)

docs = [doc1, doc2, doc3, doc4, doc5, doc6]


# # azure open ai embeding

vector_store = Chroma(
    collection_name='sample',
    persist_directory='chroma_db',
    embedding_function=AzureOpenAIEmbeddings(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    )
)

# print(vector_store)

# # using open ai 
# # vector_store = Chroma(
# #     embedding_function=OpenAIEmbeddings(),
# #     persist_directory='chroma_db',
# #     collection_name='sample'
# # )


Ids=vector_store.add_documents(docs)

data= vector_store.get(include=['embeddings','documents','metadatas'])

# print(data)


answer = vector_store.similarity_search(
    query='give me a name of adventure bike',
    k=1
)

for val in answer:
    print(val.metadata)
    print(val.metadata["example"])