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

---

# Step 2: Prepare Data

- Download and place your PDF file in a `/data` folder at the root of your project directory.
  Example path: `chroma-langchain-demo/data/document.pdf`

---

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

---

# Step 3.1: .env File

Create a `.env` file in your project root and add:

```
GOOGLE_API_KEY=your-gemini-api-key-here
```

---

# Step 4: Perform a similarity search locally

That was easy! Right? One of the benefits of Chroma is how efficient it is when handling large amounts of vector data. For this example, we're using a tiny PDF but in your real-world application, Chroma will have no problem performing these tasks on a lot more embeddings.

Let's perform a similarity search. This simply means that given a query, the database will find similar information from the stored vector embeddings. Let's see how this is done:

```python
query = "What is this document about?"
```

We can then use the similarity_search method:

```python
docs = chroma_db.similarity_search(query)
```

Another useful method is similarity_search_with_score, which also returns the similarity score represented as a decimal between 0 and 1. (1 being a perfect match).

Step 5: Query the model
We have our query and similar documents in hand. Let's send them to the large language model that we defined earlier (in Step 3) as llm.

We're going to use LangChain's RetrievalQA chain and pass in a few parameters as shown below:

```
chain = RetrievalQA.from_chain_type(llm=llm,
                                    chain_type="stuff",
                                    retriever=chroma_db.as_retriever())
response = chain(query)
```

What this does is create a chain of type stuff, use our defined llm, and our Chroma vector store as a retriever.

# step 5: expore db data

Run ChromaDB server with your data
If you haven’t already, start a ChromaDB server and set its persist_directory to your data folder:

```
chroma run
```

Connect Chroma Explorer to the server
In Chroma Explorer, set the API endpoint to your local ChromaDB server (e.g., http://localhost:8000).

Explore your database
Use the Chroma Explorer UI to browse collections, documents, and metadata.
