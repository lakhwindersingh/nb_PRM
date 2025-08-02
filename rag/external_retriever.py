# rag/external_retriever.py
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os

from langchain_huggingface import HuggingFaceEmbeddings

# Define embedding function
# def extract_embedding(document: str, model_path="BAAI/bge-small-en-v1.5"):
#     """Extract embedding using HF Embedding
#        with a default model from HF if no path is specify
#     """
#
#     embed_model = HuggingFaceEmbedding(model_name=model_path)
#     embedding = embed_model.embed_query(text=document)
#     return embedding

def get_external_vector_store(urls, index_path="data/external_index"):
    if os.path.exists(index_path):
        return FAISS.load_local(index_path, HuggingFaceEmbeddings())

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

    vector_store = FAISS.from_documents(documents, HuggingFaceEmbeddings(), index_name=index_path)
    return vector_store
