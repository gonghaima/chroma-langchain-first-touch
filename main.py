import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_API_KEY"] = ""  # Replace with your actual Gemini API key

from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

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

query = "What is this document about?"
docs = chroma_db.similarity_search(query)

chain = RetrievalQA.from_chain_type(llm=llm,
                                    chain_type="stuff",
                                    retriever=chroma_db.as_retriever())

response = chain(query)

print(response["result"])