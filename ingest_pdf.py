import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()

loader = PyPDFLoader("temp_uploaded.pdf")
pages = loader.load()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(pages, embeddings)
db.save_local("pdf_vector_store")

print("✅ Vector store saved.")
