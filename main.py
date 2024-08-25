from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


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

system_prompt = (
    "คุณเป็นแชทบทสำหรับช่วยเหลือในการตอบคำถามที่ได้รับมาจากผู้ใช้งาน "
    "ใช้ข้อมูลต่อไปนี้ในการตอบคำถาม "
    "หากไม่สามารถตอบคำถามได้ กรุณาตอบว่า \"ไม่สามารถตอบคำถามได้\" "
    "\n\n"
    "ข้อมูล: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

question = input("Please enter your question: ")
if not question:
    with open("example-question.txt", "r") as f:
        question = f.readline().strip('\n')

ret = rag_chain.invoke({"input": question})

print(f'Context is: {ret["context"]}')
print(f'Answer is: {ret["answer"]}')
