from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


from dotenv import load_dotenv
import os, traceback
from langchain_openai import AzureChatOpenAI 
from typing import TypedDict,Annotated
from langgraph.graph import StateGraph,START,END
import operator
from langchain_core.prompts import PromptTemplate, load_prompt



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


messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Tell me about lanchain."),
]

result = model.invoke(messages)
messages.append(AIMessage(content=result.content))
print("AI response:", result.content)