from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
import os
import requests

class RAGChain:
    def __init__(self, retriever, model_name="llama3.2:1b"):
        self.retriever = retriever
        
        # Detect if Ollama is available (local) or use HuggingFace (cloud)
        if self._is_ollama_available():
            from langchain_community.chat_models import ChatOllama
            self.llm = ChatOllama(model=model_name, temperature=0)
            self.llm_type = "ollama"
        else:
            # Use HuggingFace for cloud deployment
            from langchain_huggingface import HuggingFaceEndpoint
            hf_token = os.environ.get("HUGGINGFACE_API_TOKEN", "")
            if not hf_token:
                raise ValueError(
                    "Running on Streamlit Cloud requires HUGGINGFACE_API_TOKEN. "
                    "Please add it to your Streamlit secrets. "
                    "Get a free token at https://huggingface.co/settings/tokens"
                )
            self.llm = HuggingFaceEndpoint(
                repo_id="mistralai/Mistral-7B-Instruct-v0.2",
                huggingfacehub_api_token=hf_token,
                temperature=0.1,
                max_new_tokens=512
            )
            self.llm_type = "huggingface"
        
        self.chain = self._build_chain()
    
    def _is_ollama_available(self):
        """Check if Ollama is running locally"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False

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
