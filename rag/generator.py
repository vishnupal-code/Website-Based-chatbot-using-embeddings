from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
import os

class RAGChain:
    def __init__(self, retriever, model_name="llama-3.3-70b-versatile"):
        self.retriever = retriever
        
        # Always use Groq - fast and reliable
        groq_api_key = os.environ.get("GROQ_API_KEY", "")
        if not groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is required. "
                "Get your free API key at: https://console.groq.com/keys"
            )
        
        # Using Llama 3.3 70B on Groq - blazing fast inference
        self.llm = ChatGroq(
            model=model_name,
            groq_api_key=groq_api_key,
            temperature=0,
            max_tokens=512
        )
        
        self.chain = self._build_chain()

    def _build_chain(self):
        # 1. Contextualize question (for history)
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, contextualize_q_prompt
        )

        # 2. Answer question
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If the answer is not available in the context, "
            "respond exactly with: \"The answer is not available on the provided website.\"\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)

        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        return rag_chain

    def get_response(self, query: str, chat_history: list):
        """
        Generates a response for the query given the chat history.
        """
        response = self.chain.invoke({"input": query, "chat_history": chat_history})
        return response["answer"]
