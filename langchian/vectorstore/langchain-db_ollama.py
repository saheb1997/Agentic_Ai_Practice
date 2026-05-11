from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings


# steps
# 1.download Ollama with this link :-http://ollama.com/download/windows?utm_source=chatgpt.com
# 2. Then type ollam list.
# 3.Then  ollama pull llama3.1:8b

# Embedding model
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Ollama LLM
llm = OllamaLLM(model="llama3.1:8b")

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

# Vector store
vector_store = Chroma(
    collection_name="bike_collection",
    persist_directory="chroma_db",
    embedding_function=embedding
)

vector_store.add_documents(docs)

# Search
results = answer = vector_store.similarity_search(
    query='give me a name of adventure bike',
    k=1
)

for val in answer:
    print(val.metadata)
    print(val.metadata["example"])

# # LLM response
# response = llm.invoke("Explain adventure bikes")

# print(response)