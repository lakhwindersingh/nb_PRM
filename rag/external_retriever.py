import os
import logging
from typing import List, Optional
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

logger = logging.getLogger(__name__)


class ExternalDocumentRetriever:
    """Retriever for external web documents using FAISS vector store"""

    def __init__(self,
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 index_path: str = "data/external_index"):
        self.embedding_model = embedding_model
        self.index_path = index_path
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vector_store = None

        # Load existing vector store if available
        self._load_vector_store()

    def _load_vector_store(self):
        """Load existing vector store from disk"""
        if os.path.exists(self.index_path):
            try:
                self.vector_store = FAISS.load_local(
                    self.index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"Loaded existing external vector store from {self.index_path}")
            except Exception as e:
                logger.warning(f"Failed to load existing vector store: {e}")

    def add_urls(self, urls: List[str], chunk_size: int = 1000, chunk_overlap: int = 200) -> bool:
        """Add documents from URLs to the vector store"""
        documents = []

        for url in urls:
            try:
                loader = WebBaseLoader(url)
                docs = loader.load()

                # Add URL metadata
                for doc in docs:
                    doc.metadata['source_url'] = url
                    doc.metadata['retrieval_method'] = 'web_scraping'

                documents.extend(docs)
                logger.info(f"Loaded documents from {url}")

            except Exception as e:
                logger.warning(f"Failed to load {url}: {e}")

        if not documents:
            logger.error("No documents loaded from provided URLs")
            return False

        # Create or update vector store
        try:
            if self.vector_store is None:
                # Create new vector store
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
            else:
                # Add to existing vector store
                new_vector_store = FAISS.from_documents(documents, self.embeddings)
                self.vector_store.merge_from(new_vector_store)

            # Save to disk
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            self.vector_store.save_local(self.index_path)

            logger.info(f"Added {len(documents)} documents to external vector store")
            return True

        except Exception as e:
            logger.error(f"Failed to create/update vector store: {e}")
            return False

    def get_documents(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve documents similar to query"""
        if self.vector_store is None:
            logger.warning("No vector store available. Add documents first.")
            return []

        try:
            docs = self.vector_store.similarity_search(query, k=k)
            logger.info(f"Retrieved {len(docs)} external documents for query: {query}")
            return docs
        except Exception as e:
            logger.error(f"Failed to retrieve documents: {e}")
            return []

    def get_retriever(self, k: int = 5):
        """Get LangChain retriever interface"""
        if self.vector_store is None:
            raise ValueError("No vector store available. Add documents first.")
        return self.vector_store.as_retriever(search_kwargs={"k": k})


def create_external_retriever(urls: List[str],
                              index_path: str = "data/external_index") -> ExternalDocumentRetriever:
    """Convenience function to create and populate external retriever"""
    retriever = ExternalDocumentRetriever(index_path=index_path)

    if urls:
        success = retriever.add_urls(urls)
        if not success:
            raise ValueError("Failed to load documents from provided URLs")

    return retriever

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
