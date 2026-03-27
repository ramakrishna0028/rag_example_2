import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# 1. Initialize Embeddings (Must match ingestion.py)
model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# 2. Load Vector Store
collection_path = "collection"
if not os.path.exists(collection_path):
    raise FileNotFoundError(f"Vector store not found at {collection_path}. Please run ingestion.py first.")

vector_store = FAISS.load_local(
    collection_path,
    embeddings,
    allow_dangerous_deserialization=True
)

# 3. Setup LLM
# Using llama3 (already confirmed to be available via 'ollama list')
llm = OllamaLLM(model="llama3:latest")

# 4. Define Question and Prompt
query = "what is status of policies no 6"

# 5. Retrieve Context
docs = vector_store.similarity_search(query, k=6)  # Get all relevant policies
context = "\n---\n".join([doc.page_content for doc in docs])

prompt_template = PromptTemplate.from_template("""
You are a helpful assistant for ExampleTech Solutions company policies.
Use the following context to answer the user's question. 
If you don't know the answer based on the context, just say you don't know.

Context:
{context}

Question: {query}

Answer briefly and clearly:
""")

final_prompt = prompt_template.format(context=context, query=query)

# 6. Generate Answer
print(f"Querying for: '{query}'...")
response = llm.invoke(final_prompt)

print("\n--- Answer ---")
print(response)
print("--------------")