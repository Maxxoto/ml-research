from langchain.prompts import MessagesPlaceholder,HumanMessagePromptTemplate,ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory,FileChatMessageHistory
from langchain.chains import LLMChain
from dotenv import load_dotenv

chat = ChatOpenAI(temperature=0.8,model_name="gpt-4o-mini")
memory = ConversationSummaryMemory(
    # chat_memory=FileChatMessageHistory("messages.json"),
    memory_key="messages",
    return_messages=True,
    llm=chat)

prompt = ChatPromptTemplate(input_variables=["content","messages"],
                            messages=[
                                MessagesPlaceholder(variable_name="messages"),
                                HumanMessagePromptTemplate.from_template("{content}")
                                ])
chain = LLMChain(llm=chat,prompt=prompt,memory=memory)

while True:
    content = input(">> ")
    result = chain({"content":content})
    print(result)
