from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


llm = ChatOpenAI(model="gpt-4o")

text_loader = TextLoader("example-document.txt")
docs = text_loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
splitted_docs = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splitted_docs, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

contextualize_q_system_prompt = (
    "จากประวัติการสนทนาและคำถามล่าสุดของผู้ใช้ "
    "ซึ่งอาจอ้างอิงถึงบริบทในประวัติการสนทนา "
    "สร้างคำถามที่สามารถเข้าใจได้โดยไม่ต้องมีประวัติการสนทนา "
    "อนุญาตปรับรูปแบบคำถามเท่านั้น ห้ามตอบคำถาม "
    "หากไม่จำเป็นต้องปรับคำถาม ให้ส่งคำถามกลับมาในรูปแบบเดิม"
)
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
# This retriever takes "input" and optinally "chat_history" and returns list of Document.
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)
# The goal of history_aware_retriever is to retrive context based on history of the conversation.
# BUT it isn't about providing message history or "context history" at all. IT's all about getting context.

# This context is Document from history_aware_retriever.
system_prompt = (
    "คุณเป็นแชทบทสำหรับช่วยเหลือในการตอบคำถามที่ได้รับมาจากผู้ใช้งาน "
    "ใช้ข้อมูลต่อไปนี้ในการตอบคำถาม "
    "หากไม่สามารถตอบคำถามได้ กรุณาตอบว่า \"ไม่สามารถตอบคำถามได้\" "
    "\n\n"
    "ข้อมูล: {context}"
)
# This prompt include previous conversation history and input message.
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

question = input("Please enter your question: ")
if not question:
    with open("example-question.txt", "r") as f:
        question = f.readline().strip('\n')

ret = conversational_rag_chain.invoke({"input": question}, config={"configurable": {"session_id": "123"}})

print(f'Context 1 is: {ret["context"]}')
print(f'Answer 1 is: {ret["answer"]}')

ret = conversational_rag_chain.invoke({"input": "สิ่งที่เป็นหัวข้อสนทนาก่อนหน้านี้ชื่อว่าอะไร"}, config={"configurable": {"session_id": "123"}})

print(f'Context 2 is: {ret["context"]}')
print(f'Answer 2 is: {ret["answer"]}')
