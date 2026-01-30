import chromadb
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import os

class VectorStoreManager:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Initialize/Load Chroma
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name="website_content"
        )

    def add_documents(self, documents: list[Document]):
        """
        Adds documents to the vector store.
        """
        if not documents:
            return
            
        print(f"Adding {len(documents)} documents to vector store...")
        self.vector_store.add_documents(documents)
        # Chroma handles persistence automatically in recent versions

    def get_retriever(self, k=4):
        """
        Returns a retriever object.
        """
        return self.vector_store.as_retriever(search_kwargs={"k": k})

    def reset(self):
        """
        Clears the vector store (optional utility).
        """
        try:
             # This is a bit tricky with persistent client, usually deleting the collection or dir is better
            self.vector_store.delete_collection()
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name="website_content"
            )
        except Exception as e:
            print(f"Error resetting vector store: {e}")
