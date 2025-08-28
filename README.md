# Step 1: Environment Setup

1. **Grab your OpenAI or Google Gemini API Key**
2. **Set up Python application**
   - Create your project folder:
     ```bash
     mkdir chroma-langchain-demo
     ```
   - Change into the new directory and create your main Python file:
     ```bash
     cd chroma-langchain-demo
     touch main.py
     ```
   - (Optional) Create and activate your virtual environment:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
     _(On macOS, use `python3` instead of `python` for venv)_
   - Install dependencies:
     ```bash
     pip install chromadb langchain pypdf tiktoken python-dotenv langchain-google-genai
     ```

# Step 2: Prepare Data

- Download and place your PDF file in a `/data` folder at the root of your project directory.
  Example path: `chroma-langchain-demo/data/document.pdf`

# Step 3: Example Usage in `main.py`

```python
from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma

loader = PyPDFLoader("data/document.pdf")
docs = loader.load_and_split()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.environ["GOOGLE_API_KEY"]
)
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=os.environ["GOOGLE_API_KEY"])

chroma_db = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="data",
    collection_name="lc_chroma_demo"
)
```

# Step 4: .env File

Create a `.env` file in your project root and add:
```
GOOGLE_API_KEY=your-gemini-api-key-here
```

# Step 5: .gitignore

Add `.env` to your `.gitignore` to keep your API key safe:
```
.env
```

---

This setup uses Google Gemini for both LLM and embeddings. For OpenAI, update the imports and API key accordingly.

