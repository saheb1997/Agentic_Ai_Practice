import os
from dotenv import load_dotenv
from langchain_community.llms.ollama import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Set env vars (fixing typos in variable names)
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# Prompt setup
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond."),
        ("user", "Question: {question}")
    ]
)

# Streamlit interface
st.title("LangChain Demo with LLaMA2")
input_text = st.text_input("What question do you have in mind?")

# LangChain model and chain
llm = Ollama(model="llama2")  
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    response = chain.invoke({"question": input_text})
    st.write(response)
