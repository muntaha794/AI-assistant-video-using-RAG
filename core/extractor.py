from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os 

def get_llm():
    return ChatMistralAI(model = "mistral-small-latest", mistral_api_key = os.getenv("MISTRAL_API_KEY"),temperature=0.3)

def build_chain(system_prompt : str):
    llm = get_llm()
    return (RunnablePassthrough() | RunnableLambda(lambda x:{"text":x}) | ChatPromptTemplate.from_messages(
        [
        ("system", system_prompt),
        ("human", "{text}"),
    ]
    ) | llm | StrOutputParser()
    )

def extract_action_items(transcript : str) -> str:
    chain = build_chain(
        "You are an expert meeting assistant. From the meeting transcript. Extract all action items from the following meeting transcript. for each provide:\n"
        "_ Task description\n"
        " _Owner (who is responsible for completing the task)\n"
        "_ Deadline (if mentioned else write 'Not specified')\n"
        "Format as a numbered list. If no action items are found, respond with 'No action items found.'"
    )

    return chain.invoke(transcript)

    prompt = ChatPromptTemplate.from_messages(      
        [
        ("system", system_prompt),
        ("human", "{text}"),
    ]
    )

    chain = RunnablePassthrough() | RunnableLambda(lambda x:{"text":x}) | prompt | llm | StrOutputParser()

    return chain