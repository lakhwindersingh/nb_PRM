# rag/external_retriever.py
from langchain.document_loaders import WebBaseLoader
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

def get_external_vector_store(urls, index_path="data/external_index"):
    if os.path.exists(index_path):
        return FAISS.load_local(
            index_path,
            HuggingFaceEmbeddings(),
            allow_dangerous_deserialization=True  # Add this flag
        )

    # Load and embed external documents
    documents = []
    for url in urls:
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            print(f"[Warning] Failed to load {url}: {e}")

    if not documents:
        raise ValueError("No external documents loaded")

    # Create FAISS vector store from documents
    vector_store = FAISS.from_documents(documents, HuggingFaceEmbeddings())

    # Save the vector store to disk
    vector_store.save_local(index_path)

    return vector_store
