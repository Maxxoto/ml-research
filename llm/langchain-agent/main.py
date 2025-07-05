from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor, create_openai_functions_agent
from dotenv import load_dotenv
from tools.sql import run_query_tool

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.8)

prompt = ChatPromptTemplate.from_messages(
    messages=[HumanMessagePromptTemplate.from_template("{input}"),
              MessagesPlaceholder(variable_name="agent_scratchpad")])

agent = create_openai_functions_agent(
    llm=llm, tools=[run_query_tool], prompt=prompt)

agent_executor = AgentExecutor(
    agent=agent, tools=[run_query_tool], verbose=True)

agent_executor.invoke(
    {'input': "How many users are provided a shipping address in address table?"})
