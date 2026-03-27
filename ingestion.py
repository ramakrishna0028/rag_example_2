import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Load the PDF document
pdf_path = os.path.join("data", "sample_company_policies_rag.pdf")
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found at {pdf_path}")

print(f"Loading PDF: {pdf_path}")
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# 2. Split text into chunks
# Increased chunk_size to 500 for better semantic context
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
splits = text_splitter.split_documents(documents)
print(f"Created {len(splits)} chunks from {len(documents)} pages.")

# 3. Initialize Embeddings
# Using langchain_huggingface for modern support
model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# 4. Create and Save Vector Store
print("Generating embeddings and creating FAISS index...")
vector_store = FAISS.from_documents(splits, embeddings)

# Save to local directory
save_dir = "collection"
vector_store.save_local(save_dir)
print(f"FAISS index saved to '{save_dir}' directory.")