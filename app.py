import streamlit as st
import os
from crawler.scraper import scrape_website, chunk_text
from vector_db.store import VectorStoreManager
from rag.generator import RAGChain
from langchain_core.messages import HumanMessage, AIMessage

# Page Config
st.set_page_config(page_title="Website Chatbot", layout="wide")
st.title("🌐 Website-Based Chatbot")

# Display LLM info in sidebar
st.sidebar.success("🚀 Powered by Groq (Llama 3.3 70B)")
st.sidebar.caption("Fast AI inference with LPU technology")

# Session State Initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store_manager" not in st.session_state:
    st.session_state.vector_store_manager = VectorStoreManager()
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "indexed" not in st.session_state:
    st.session_state.indexed = False
# Clear RAG chain if environment changed (helps with cloud deployment)
if "llm_env" not in st.session_state:
    st.session_state.llm_env = "cloud" if os.environ.get("HUGGINGFACE_API_TOKEN") else "local"
else:
    current_env = "cloud" if os.environ.get("HUGGINGFACE_API_TOKEN") else "local"
    if st.session_state.llm_env != current_env:
        st.session_state.rag_chain = None
        st.session_state.llm_env = current_env

# Sidebar
with st.sidebar:
    st.header("Configuration")
    url_input = st.text_input("Enter Website URL", placeholder="https://example.com")
    
    if st.button("Index Website"):
        if not url_input:
            st.error("Please enter a valid URL.")
        else:
            with st.spinner("Crawling and Indexing..."):
                try:
                    # Clear previous website data
                    st.session_state.vector_store_manager.reset()
                    
                    # Scrape
                    text = scrape_website(url_input)
                    if text:
                        # Chunk
                        chunks = chunk_text(text, source=url_input)
                        st.info(f"Extracted {len(chunks)} chunks.")
                        
                        # Store
                        st.session_state.vector_store_manager.add_documents(chunks)
                        
                        # Initialize RAG Chain
                        retriever = st.session_state.vector_store_manager.get_retriever()
                        st.session_state.rag_chain = RAGChain(retriever)
                        
                        # Clear chat history for new website
                        st.session_state.chat_history = []
                        
                        st.session_state.indexed = True
                        st.success("Website indexed successfully! Chat history cleared.")
                    else:
                        st.error("Failed to extract content.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

    if st.button("Clear Memory"):
        st.session_state.chat_history = []
        st.success("Conversation history cleared.")

# Main Chat Interface
if st.session_state.indexed:
    # Display Chat History
    for message in st.session_state.chat_history:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(message.content)

    # User Input
    if prompt := st.chat_input("Ask a question about the website content..."):
        # Display User Message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate Response
        with st.chat_message("assistant"):
            if st.session_state.rag_chain:
                with st.spinner("Thinking..."):
                    response = st.session_state.rag_chain.get_response(
                        prompt, 
                        st.session_state.chat_history
                    )
                    st.markdown(response)
                    
                    # Update History
                    st.session_state.chat_history.append(HumanMessage(content=prompt))
                    st.session_state.chat_history.append(AIMessage(content=response))
            else:
                 st.error("RAG Chain not initialized. Please re-index.")

else:
    st.info("Please enter a URL in the sidebar and click 'Index Website' to start chatting.")
