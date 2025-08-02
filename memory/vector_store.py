import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings


def get_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    faiss_path = "faiss_index"

    # if os.path.exists(faiss_path):
    #     return FAISS.load_local(faiss_path, embeddings)
    # else:
    dummy_text = [""]  # A placeholder to avoid empty list issues
    vector_store = FAISS.from_texts(texts=dummy_text, embedding=embeddings)
    # vector_store.save_local(faiss_path)
    return vector_store
