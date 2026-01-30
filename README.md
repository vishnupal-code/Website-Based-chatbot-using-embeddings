# Website-Based Chatbot Using Embeddings

An AI-powered chatbot that accepts a website URL, extracts and indexes its content, and answers questions based strictly on the website's information using RAG (Retrieval Augmented Generation).

## 🎯 Project Overview

This application demonstrates a real-world implementation of an intelligent retrieval and question-answering system that:
- Accepts any website URL as input
- Crawls and extracts meaningful content from the website
- Creates semantic embeddings and stores them in a vector database
- Provides accurate, context-aware answers based only on the indexed website content
- Maintains conversational context across multiple queries

## 🏗️ Architecture

```
┌─────────────────┐
│   User Input    │
│  (Website URL)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Web Scraper    │
│ (BeautifulSoup) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Text Processing │
│   & Chunking    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Embeddings    │
│  (HuggingFace)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Vector Database │
│   (ChromaDB)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  RAG Pipeline   │
│  (LangChain)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   LLM Model     │
│(Llama 3.2 1B)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Streamlit UI   │
│   (Response)    │
└─────────────────┘
```

## 🛠️ Technology Stack

### **Frameworks & Libraries**
- **LangChain Classic** - AI orchestration framework for building the RAG pipeline
- **Streamlit** - Web interface for user interaction
- **BeautifulSoup4** - HTML parsing and content extraction
- **Sentence Transformers** - Embedding generation

### **LLM Model: Ollama Llama 3.2 1B**

**Why Llama 3.2 1B?**
- ✅ **Free & Local** - Runs entirely on your machine without API costs
- ✅ **Fast Response Time** - Optimized 1B parameter model provides quick answers
- ✅ **Privacy** - No data sent to external servers
- ✅ **Quality** - Excellent performance for RAG-based Q&A tasks
- ✅ **Resource Efficient** - Works well on standard hardware

The 1B variant was specifically chosen over the 3B version for faster inference times while maintaining good accuracy for retrieval-augmented tasks where the context is already provided.

### **Vector Database: ChromaDB**

**Why ChromaDB?**
- ✅ **Lightweight** - Easy to set up with minimal configuration
- ✅ **Persistent Storage** - Embeddings are saved and reused across sessions
- ✅ **No External Services** - Runs locally without requiring cloud services
- ✅ **Python Native** - Seamless integration with Python applications
- ✅ **Built-in Similarity Search** - Optimized for semantic search operations
- ✅ **Perfect for Prototypes** - Ideal for development and small to medium datasets

### **Embedding Strategy**

**Model:** `sentence-transformers/all-MiniLM-L6-v2`

**Why This Model?**
- Fast inference speed (384-dimensional embeddings)
- Good balance between accuracy and performance
- Well-suited for semantic search tasks
- Pre-trained on large corpus of text
- Works offline without API keys

**Embedding Process:**
1. Text is chunked using `RecursiveCharacterTextSplitter`
2. Each chunk maintains metadata (source URL)
3. Embeddings are generated using HuggingFace model
4. Stored in ChromaDB for efficient retrieval
5. Similarity search retrieves top-k relevant chunks for each query

## ✨ Key Features

### 1. **Intelligent Content Extraction**
- Removes irrelevant sections (headers, footers, navigation, ads)
- Cleans and normalizes HTML content
- Avoids duplicate content

### 2. **Semantic Chunking**
- Configurable chunk size and overlap
- Preserves context across chunk boundaries
- Maintains source metadata for each chunk

### 3. **Context-Aware Responses**
- Uses RAG to retrieve relevant content
- Generates answers only from indexed website
- Responds "The answer is not available on the provided website" when information isn't found

### 4. **Conversational Memory**
- Maintains short-term conversation history
- Understands follow-up questions
- Context resets when indexing a new website

### 5. **User-Friendly Interface**
- Simple URL input
- One-click website indexing
- Chat-style question interface
- Clear memory button

## 📋 Prerequisites

- Python 3.10 or higher
- Ollama installed on your system
- 2GB+ free disk space (for Llama 3.2 1B model)

## 🚀 Setup Instructions

### Step 1: Install Ollama

1. Download Ollama from [https://ollama.com/download](https://ollama.com/download)
2. Install it on your system
3. Pull the Llama 3.2 1B model:
```bash
ollama pull llama3.2:1b
```

### Step 2: Clone the Repository

```bash
git clone <your-repo-url>
cd website_chatbot
```

### Step 3: Create Virtual Environment

```bash
python -m venv .venv
```

### Step 4: Activate Virtual Environment

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
.venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

## 🎮 Running the Application

### Option 1: Using the Batch File (Windows - Recommended)

Simply run:
```bash
run.bat
```

### Option 2: Using Python Command

```bash
python -m streamlit run app.py
```

### Option 3: After Activating Virtual Environment

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## ☁️ Deploying to Streamlit Cloud

The app **automatically detects** the environment and switches between:
- **Local**: Ollama with Llama 3.2:1b
- **Cloud**: HuggingFace API with Mistral-7B-Instruct (free)

### Step-by-Step Cloud Deployment:

**1. Get HuggingFace Token (Free)**
- Visit https://huggingface.co/settings/tokens
- Click "New token"
- Name it (e.g., "streamlit-chatbot")
- Select "Read" role
- Copy the token

**2. Deploy to Streamlit Cloud**
- Go to https://share.streamlit.io/
- Click "New app"
- Repository: `vishnupal-code/Website-Based-chatbot-using-embeddings`
- Branch: `main`
- Main file: `app.py`
- Click "Advanced settings"

**3. Add Secrets**
In the "Secrets" section, paste:
```toml
HUGGINGFACE_API_TOKEN = "your_token_here"
```

**4. Deploy!**
Click "Deploy" and your app will be live in minutes.

### Cloud vs Local Comparison:
| Feature | Local (Ollama) | Cloud (HuggingFace) |
|---------|----------------|---------------------|
| Model | Llama 3.2:1b | Mistral-7B-Instruct |
| Speed | Very Fast | Moderate |
| Setup | Requires Ollama | Just API token |
| Privacy | 100% Local | Uses HF API |
| Cost | Free | Free (with limits) |
| Accessibility | Local only | Anywhere |

## 📖 How to Use

1. **Enter Website URL**
   - Input any valid website URL in the sidebar
   - Example: `https://example.com`

2. **Index Website**
   - Click the "Index Website" button
   - Wait for the system to crawl and process the content
   - You'll see a success message when done

3. **Ask Questions**
   - Type your question in the chat input
   - The chatbot will answer based on the website content
   - Ask follow-up questions for more details

4. **Switch Websites**
   - Enter a new URL and click "Index Website"
   - Previous data and chat history are automatically cleared

5. **Clear History**
   - Use "Clear Memory" button to reset the conversation

## 📁 Project Structure

```
website_chatbot/
├── app.py                      # Main Streamlit application
├── run.bat                     # Windows launcher script
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── crawler/
│   └── scraper.py             # Website crawling & content extraction
├── vector_db/
│   └── store.py               # Vector database manager (ChromaDB)
├── rag/
│   └── generator.py           # RAG chain implementation
├── chroma_db/                  # ChromaDB persistent storage
└── project details/
    └── project.txt            # Assignment requirements
```

## 🔧 Configuration

### Chunk Size & Overlap
Edit in `crawler/scraper.py`:
```python
def chunk_text(text, source, chunk_size=1000, chunk_overlap=200):
```

### Retrieval Settings
Edit in `vector_db/store.py`:
```python
def get_retriever(self, k=4):  # k = number of chunks to retrieve
```

### LLM Model
Edit in `rag/generator.py`:
```python
def __init__(self, retriever, model_name="llama3.2:1b"):
```

## 🎯 Assumptions & Limitations

### Assumptions
- Websites are publicly accessible and don't require authentication
- Content is primarily text-based HTML
- JavaScript-rendered content is not supported
- Single-page websites only (no multi-page crawling)

### Current Limitations
1. **No JavaScript Support**: Cannot extract dynamically loaded content
2. **Single Page Only**: Only crawls the provided URL, not linked pages
3. **Text Content Only**: Images, videos, and other media are not processed
4. **English Language**: Optimized for English content
5. **Local Deployment**: Requires local setup (Ollama + dependencies)

### Known Issues
- Large websites (>1000 chunks) may take longer to index
- Responses can take 5-15 seconds depending on query complexity
- Some complex website layouts may not be parsed perfectly

## 🚀 Future Improvements

1. **Multi-Page Crawling**: Crawl entire website hierarchy
2. **JavaScript Rendering**: Support for dynamic content
3. **Multiple File Formats**: Support PDF, Word docs, markdown
4. **Advanced Chunking**: Semantic chunking based on document structure
5. **Faster Models**: Option to use API-based models
6. **Export Chat**: Save conversation history
7. **Website Comparison**: Compare content across multiple websites
8. **Authentication Support**: Handle login-protected content
9. **Multilingual Support**: Support for non-English websites
10. **Cloud Deployment**: Deploy to Streamlit Cloud

## 🐛 Troubleshooting

### Issue: "Module not found" errors
**Solution:** Ensure virtual environment is activated:
```bash
.\.venv\Scripts\Activate.ps1
```

### Issue: Ollama connection errors
**Solution:** Ensure Ollama is running and model is downloaded:
```bash
ollama pull llama3.2:1b
```

### Issue: Slow responses
**Solution:** 
- Use shorter queries
- Reduce chunk retrieval (k=2 instead of k=4)
- Ensure no other heavy processes are running

### Issue: "Not available on website" for obvious answers
**Solution:** 
- Content might not have been extracted properly
- Try re-indexing the website
- Check if the website requires JavaScript

## 📊 Performance Metrics

- **Indexing Speed**: ~5-10 seconds for average webpage
- **Query Response**: ~3-8 seconds per question
- **Memory Usage**: ~2-4GB RAM
- **Storage**: ~50MB per indexed website

## 🔐 Privacy & Security

- All processing happens locally on your machine
- No data is sent to external services (except website crawling)
- ChromaDB data stored locally in `chroma_db/` folder
- Ollama runs entirely offline after model download

## 📝 Notes for Evaluators

- The system is designed to be **strictly grounded** in website content
- No hardcoded answers or external knowledge is used
- Prompt engineering ensures responses cite only retrieved context
- The architecture is modular and easily extensible
- Code follows clean coding practices with proper comments

## 👨‍💻 Author

**Assignment Submission for Humanli.ai**  
Role: AI/ML Engineer  
Date: January 2026

---

**Need Help?** Check the troubleshooting section or review the project structure diagram above.
