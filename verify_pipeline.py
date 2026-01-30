import os
from crawler.scraper import scrape_website, chunk_text
from vector_db.store import VectorStoreManager
from rag.generator import RAGChain
import sys

def test_pipeline():
    print("--- Starting Verification Script ---")
    
    # 1. Test Scraping
    url = "https://example.com"
    print(f"1. Testing Crawler on {url}...")
    try:
        text = scrape_website(url)
        if "Example Domain" in text:
            print("   [PASS] Content extracted successfully.")
        else:
            print(f"   [WARN] Extracted text might be missing expected content: {text[:50]}...")
    except Exception as e:
        print(f"   [FAIL] Crawler Exception: {e}")
        return

    # 2. Test Chunking
    print("2. Testing Chunking...")
    try:
        chunks = chunk_text(text, source=url)
        print(f"   [PASS] Created {len(chunks)} chunks.")
    except Exception as e:
        print(f"   [FAIL] Chunking Exception: {e}")
        return

    # 3. Test Vector Store
    print("3. Testing Vector Store (ChromaDB)...")
    try:
        store = VectorStoreManager(persist_directory="./test_chroma_db")
        store.add_documents(chunks)
        retriever = store.get_retriever()
        print("   [PASS] Vector Store initialized and documents added.")
    except Exception as e:
        print(f"   [FAIL] Vector Store Exception: {e}")
        return

    # 4. Test RAG Chain
    print("4. Testing RAG Chain (Ollama)...")
    try:
        rag = RAGChain(retriever, model_name="mistral")
        # Check if Ollama is running effectively by a dry run or mocking if needed. 
        # Assuming Ollama is running for this test.
        print("   [INFO] Initialized RAG Chain.")
        
        # We won't invoke the LLM here to avoid hanging if Ollama isn't up, 
        # but the initialization confirms imports and class structure.
        print("   [PASS] RAG Chain instantiation successful.")
        
    except Exception as e:
        print(f"   [FAIL] RAG Chain Exception (Is Ollama running?): {e}")

    print("--- Verification Complete ---")

if __name__ == "__main__":
    test_pipeline()
