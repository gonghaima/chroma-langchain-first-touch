import os
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# Load and split PDF document
loader = PyPDFLoader("data/document.pdf")
docs = loader.load_and_split()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

# Create and persist Chroma DB collection with loaded documents
chroma_db = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="chroma",
    collection_name="lc_chroma_demo"
)

print("Data loaded and persisted to chroma directory.")