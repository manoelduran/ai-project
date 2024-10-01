from scrapper import get_text_from_url
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

def get_response_from_openai(message: str):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    response = llm.invoke(message)
    return response

def documentation_tool(url:str , question: str) -> str:
    """ This tool receives as input the URL from the documentation and the question about the documentation that the user wants to be answered  """
    context = get_text_from_url(url)

    messages = [
        SystemMessage(content="You're a helpful programming assistant that must explain programming library documentations to users as simples as possible"),
        HumanMessage(content=f"Documentation: {context} \n\n {question}")
    ]
    response = get_response_from_openai(messages)
    return response