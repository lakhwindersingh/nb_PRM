# rag/external_retriever.py
from typing import List

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import os

from langchain_core.retrievers import BaseRetriever

class SimpleRetriever(BaseRetriever):
    docs: List[Document]
    k: int = 5

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Return the first k documents from the list of documents"""
        return self.docs[:self.k]

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """(Optional) async native implementation."""
        return self.docs[:self.k]

class CompositeRetriever:
    def __init__(self, internal_retriever: BaseRetriever, external_vector_store: BaseRetriever):
        self.internal_retriever = internal_retriever
        self.external_retriever = external_vector_store.as_retriever()

    def get_combined_context(self, query, k=4):
        internal_docs = self.internal_retriever.get_relevant_documents(query)
        external_docs = self.external_retriever.get_relevant_documents(query)

        all_docs = internal_docs + external_docs
        deduped = list({doc.page_content: doc for doc in all_docs}.values())

        combined_text = "\n".join(doc.page_content.strip() for doc in deduped[:k])
        return combined_text

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
