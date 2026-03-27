# Diagnostic cell — run this in one cell to confirm envs and test AzureChatOpenAI
from dotenv import load_dotenv
import os, traceback
from langchain_openai import AzureChatOpenAI 
from typing import TypedDict,Annotated
from langgraph.graph import StateGraph,START,END
import operator
from langchain_core.prompts import PromptTemplate



load_dotenv()  # ensure .env loaded

print("AZURE_OPENAI_KEY present:", bool(os.getenv("AZURE_OPENAI_KEY")))
print("AZURE_OPENAI_ENDPOINT:", os.getenv("AZURE_OPENAI_ENDPOINT"))
print("AZURE_OPENAI_DEPLOYMENT:", os.getenv("AZURE_OPENAI_DEPLOYMENT"))
print("AZURE_OPENAI_API_VERSION:", os.getenv("AZURE_OPENAI_API_VERSION"))
print("OPENAI_API_KEY (if set):", bool(os.getenv("OPENAI_API_KEY")))

# Map env names the SDK might look for (non-destructive)
if os.getenv("AZURE_OPENAI_KEY") and not os.getenv("AZURE_OPENAI_API_KEY"):
    os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_KEY")
if os.getenv("AZURE_OPENAI_KEY") and not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_KEY")

# Instantiate explicitly with api_key to avoid env name issues
try:
    model = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        temperature=0.2,
        api_key=os.getenv("AZURE_OPENAI_KEY"),  # explicit
    )
    print('AzureChatOpenAI instantiated OK')
except Exception as e:
    print('Failed to instantiate AzureChatOpenAI:')
    traceback.print_exc()

# Try a short call (small tokens) to test end-to-end
try:
    resp = model.invoke("Say hello in one short sentence.")
    print("Invoke result:", getattr(resp, "content", None) or resp)
except Exception as e:
    print('Error during invoke:')
    traceback.print_exc()





import streamlit as st
load_dotenv()

st.header("Research Tool")

paper_input = st.selectbox( "Select Research Paper Name", ["Select...", "Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] ) 

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "CodeOriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )


#template
template = PromptTemplate(
    template = """
                Please summarize the research paper titled "{paper_input}" with the following specifications: Explanation Style: {style_input} Explanation Length: {length_input} 1. Mathematical Details: - Include relevant mathematical equations if present in the paper. - Explain the mathematical concepts using simple, intuitive code snippets where applicable. 2. Analogies: - Use relatable analogies to simplify complex ideas. If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing. Ensure the summary is clear, accurate, and aligned with the provided style and length.
""",
            input_variables = ['paper_input','style_input','length_input']
)


promt= template.invoke(
    {
        'paper_input':paper_input,
        'style_input':style_input,
        'length_input':length_input
    }
)
if st.button('Summarize'):
    result = model.invoke(promt)
    st.write(result.content)
    
    # result  = model.invoke(user_imput)
    # st.write(result.content)