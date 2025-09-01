import os
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Load and split PDF document
loader = PyPDFLoader("data/document.pdf")
docs = loader.load_and_split()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.environ["GOOGLE_API_KEY"]
)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=os.environ["GOOGLE_API_KEY"])

# Load existing Chroma DB collection
chroma_db = Chroma(
    persist_directory="data",
    embedding_function=embeddings,
    collection_name="lc_chroma_demo"
)

query = "What is this document about?"
docs = chroma_db.similarity_search(query)

chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=chroma_db.as_retriever()
)

response = chain.invoke(query)
print(response["result"])