from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


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

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# the reason we pass string directly to the chain is because this chain expect a string input
# the dict of "context" and "input" is converted to RunnableParallel.
# RunnableParallel returns a dict of "context" and "input" which is then passed to the prompt
print(rag_chain.invoke("ช่วยบอกเกี่ยวกับบริษัท CJ More หน่อยครับ"))
