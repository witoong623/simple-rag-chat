from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import TextLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Config


class Chatbot:
    ''' This class can't be serialized because one of langchain objects can't be serialized '''
    def __init__(self, config: Config):
        self.config = config

        chat_kwargs = {
            'openai_api_key': config.openai_api_key,
        }
        if config.openai_base_url:
            chat_kwargs['base_url'] = config.openai_base_url
        self.llm = ChatOpenAI(**chat_kwargs)

        self.document_path = config.document_path
        self.store = {}
        self.setup_retriever()
        self.setup_chain()

    def setup_retriever(self):
        text_loader = TextLoader(self.document_path)
        docs = text_loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=200,
        )
        splitted_docs = text_splitter.split_documents(docs)

        embedding_kwargs = {
            'openai_api_key': self.config.openai_api_key,
        }
        if self.config.openai_base_url:
            embedding_kwargs['base_url'] = self.config.openai_base_url
        # embedding need to be from llama.cpp server at /embedding too
        vectorstore = Chroma.from_documents(documents=splitted_docs,
                                            embedding=OpenAIEmbeddings(**embedding_kwargs))
        self.retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    def setup_chain(self):
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
            self.llm, self.retriever, contextualize_q_prompt
        )
        # The goal of history_aware_retriever is to retrive context based on history of the conversation.
        # BUT it isn't about providing message history or "context history" at all. IT's all about getting context.

        # This context is Document from history_aware_retriever.
        qa_system_prompt = (
            "คุณเป็นแชทบทสำหรับช่วยเหลือในการตอบคำถามที่ได้รับมาจากผู้ใช้งาน "
            "ใช้ข้อมูลต่อไปนี้ในการตอบคำถาม "
            "หากไม่สามารถตอบคำถามได้ กรุณาตอบว่า \"ไม่สามารถตอบคำถามได้\" "
            "\n\n"
            "ข้อมูล: {context}"
        )
        # This prompt include input, previous conversation history and input message.
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        qa_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

        self.conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def chat(self, question, session_id="123"):
        return self.conversational_rag_chain.stream(
            {"input": question},
            config={"configurable": {"session_id": session_id}}
        )
