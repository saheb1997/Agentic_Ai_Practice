from dotenv import load_dotenv
import os, traceback
from langchain_openai import AzureChatOpenAI 
from typing import TypedDict,Annotated,Literal,Optional
from langgraph.graph import StateGraph,START,END
import operator
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser,StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()  # ensure .env loaded

# Instantiate explicitly with api_key to avoid env name issues

model = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        temperature=0.2,
        api_key=os.getenv("AZURE_OPENAI_KEY"),  # explicit
    )


prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text\n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(

    template='Generate 5 short question aswers from the follwoing text\n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes -> {notes} and quize ->{quize}',
    input_variables= ['notes','quize']
)

parser = StrOutputParser()


parallel_chain = RunnableParallel(
    {
        'notes': prompt1| model | parser,
        'quize' : prompt2 | model | parser
    }
)

merge_chain = prompt3 |model | parser

chain = parallel_chain | merge_chain

text ="Scikit-learn defines a simple API for creating visualizations for machine learning. The key feature of this API is to allow for quick plotting and visual adjustments without recalculation. We provide Display classes that expose two methods for creating plots: from_estimator and from_predictions.The from_estimator method generates a Display object from a fitted estimator, input data (X, y), and a plot. The from_predictions method creates a Display object from true and predicted values (y_test, y_pred), and a plot.Using from_predictions avoids having to recompute predictions, but the user needs to take care that the prediction values passed correspond to the pos_label. For predict_proba, select the column corresponding to the pos_label class while for decision_function, revert the score (i.e. multiply by -1) if pos_label is not the last class in the classes_ attribute of your estimator.The Display object stores the computed values (e.g., metric values or feature importance) required for plotting with Matplotlib. These values are the results derived from the raw predictions passed to from_predictions, or an estimator and X passed to from_estimator.Display objects have a plot method that creates a matplotlib plot once the display object has been initialized (note that we recommend that display objects are created via from_estimator or from_predictions instead of initialized directly). The plot method allows adding to an existing plot by passing the existing plots matplotlib.axes.Axes to the ax parameter.In the following example, we plot a ROC curve for a fitted Logistic Regression model from_estimator"

result = chain.invoke({'text':text})
print(result)