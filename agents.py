import os
from scrapper import get_text_from_url
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool

api_key = os.getenv("OPENAI_API_KEY")

def get_response_from_openai(message: str):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    response = llm.invoke(message)
    return response

@tool
def documentation_tool(url:str , question: str) -> str:
    """ This tool receives as input the URL from the documentation and the question about the documentation that the user wants to be answered  """
    context = get_text_from_url(url)

    messages = [
        SystemMessage(content="You're a helpful programming assistant that must explain programming library documentations to users as simple as possible."),
        HumanMessage(content=f"Documentation: {context} \n\n {question}")
    ]
    response = get_response_from_openai(messages)
    return response

@tool
def black_formatter_tool(path: str) -> str:
    """ This tool receives as input a file system path to a python script file and runs black formatter to format the file's python code """
    try:
        os.system(f"poetry run black {path}")
        return "Done!"
    except Exception as e:
        return f"Did not work. Error: {str(e)}"

toolkit = [documentation_tool, black_formatter_tool]

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=api_key)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
            You are a programming assistant. Use your tools to answer questions.
         If you don't have a tool to answer the question, say so.
         Return only the answers.
        """),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad")
    ]
)
agent = create_openai_tools_agent(llm, toolkit, prompt)
agent_executor = AgentExecutor(agent=agent, tools=toolkit, verbose=True)

# Example call
result = agent_executor.invoke({"input": "Explain me the documentation of this library: https://docs.python.org/3/library/os.html"})
print(result["output"])
