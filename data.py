import os
from langchain.vectorstores.chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceEmbeddings

CHROMA_PATH = "./chroma"
PDF_PATH = "./data.pdf"

# Step 1: Extract text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        print("reading page")
        text += page.extract_text()
    return text

# Step 2: Split text into chunks
def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

# Step 3: Create Chroma Vector Store from PDF
def create_chroma_db_from_pdf(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    chunks = split_text(text)
    documents = [Document(page_content=chunk) for chunk in chunks]

    db = Chroma.from_documents(documents, HuggingFaceEmbeddings(), persist_directory=CHROMA_PATH)
    return db

# Check if Chroma DB exists, otherwise create it from PDF
if not os.path.exists(CHROMA_PATH):
    os.makedirs(CHROMA_PATH)
    create_chroma_db_from_pdf(PDF_PATH)
