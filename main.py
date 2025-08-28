import os
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