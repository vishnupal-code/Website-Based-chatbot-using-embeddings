import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import re

def scrape_website(url: str) -> str:
    """
    Scrapes the content from the given URL, removing irrelevant sections.
    """
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Remove irrelevant elements
        for script in soup(["script", "style", "header", "footer", "nav", "aside", "form"]):
            script.decompose()

        # Get text
        text = soup.get_text(separator=" ")
        
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        
        if not text:
            raise ValueError("No meaningful content extracted.")
            
        return text

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""

def chunk_text(text: str, source: str) -> list[Document]:
    """
    Splits the text into chunks and adds metadata.
    """
    if not text:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks = text_splitter.create_documents([text], metadatas=[{"source": source}])
    return chunks
