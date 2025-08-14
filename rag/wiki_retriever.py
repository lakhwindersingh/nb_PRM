# rag/wikipedia_retriever.py
import os
import pickle
import hashlib
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple

import faiss
import numpy as np
import wikipediaapi
import wikipedia
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from langchain_community.retrievers import WikipediaRetriever as LangChainWikipediaRetriever
from langchain_community.utilities import WikipediaAPIWrapper

logger = logging.getLogger(__name__)


class WikipediaVectorRetriever:
    """High-performance Wikipedia retriever using FAISS for semantic search"""

    def __init__(self,
                 model_name: str = 'all-MiniLM-L6-v2',
                 cache_dir: str = "data/wikipedia_cache",
                 index_path: str = "data/wikipedia_faiss_index",
                 max_cache_age_days: int = 7):

        self.model = SentenceTransformer(model_name)
        self.cache_dir = cache_dir
        self.index_path = index_path
        self.max_cache_age = timedelta(days=max_cache_age_days)

        # Ensure directories exist
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(os.path.dirname(index_path), exist_ok=True)

        # Initialize Wikipedia API
        self.wiki_api = wikipediaapi.Wikipedia(
            language='en',
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            user_agent="WikipediaRetriever/1.0 (https://example.com/contact)"
        )

        # Initialize components
        self.index = None
        self.sentences = []
        self.sentence_metadata = []

        # Load existing index if available
        self._load_or_create_index()

    def _load_or_create_index(self):
        """Load existing FAISS index or create new one"""
        if os.path.exists(f"{self.index_path}.faiss") and os.path.exists(f"{self.index_path}_metadata.pkl"):
            try:
                self._load_index()
                logger.info(f"Loaded existing FAISS index with {len(self.sentences)} sentences")
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}. Creating new one.")
                self._create_empty_index()
        else:
            self._create_empty_index()

    def _create_empty_index(self):
        """Create empty FAISS index"""
        d = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(d)
        self.sentences = []
        self.sentence_metadata = []

    def _load_index(self):
        """Load FAISS index and metadata from disk"""
        self.index = faiss.read_index(f"{self.index_path}.faiss")

        with open(f"{self.index_path}_metadata.pkl", 'rb') as f:
            data = pickle.load(f)
            self.sentences = data['sentences']
            self.sentence_metadata = data['metadata']

    def _save_index(self):
        """Save FAISS index and metadata to disk"""
        faiss.write_index(self.index, f"{self.index_path}.faiss")

        with open(f"{self.index_path}_metadata.pkl", 'wb') as f:
            pickle.dump({
                'sentences': self.sentences,
                'metadata': self.sentence_metadata
            }, f)

    def add_wikipedia_page(self, page_title: str) -> bool:
        """Add a Wikipedia page to the index"""
        try:
            page = self.wiki_api.page(page_title)
            if not page.exists():
                logger.warning(f"Wikipedia page '{page_title}' does not exist")
                return False

            # Split content into sentences/paragraphs
            text_content = page.text
            sentences = self._split_content(text_content)

            if not sentences:
                logger.warning(f"No content extracted from page '{page_title}'")
                return False

            # Generate embeddings
            embeddings = self.model.encode(sentences, normalize_embeddings=True)

            # Add to index
            self.index.add(embeddings)

            # Store sentences and metadata
            for i, sentence in enumerate(sentences):
                self.sentences.append(sentence)
                self.sentence_metadata.append({
                    'page_title': page.title,
                    'page_url': page.fullurl,
                    'sentence_index': i,
                    'timestamp': datetime.now().isoformat()
                })

            logger.info(f"Added {len(sentences)} sentences from '{page_title}' to index")
            return True

        except Exception as e:
            logger.error(f"Failed to add Wikipedia page '{page_title}': {e}")
            return False

    def _split_content(self, text: str) -> List[str]:
        """Split text into meaningful chunks"""
        # Split by paragraphs first
        paragraphs = text.split('\n\n')

        chunks = []
        for para in paragraphs:
            para = para.strip()
            if len(para) < 50:  # Skip very short paragraphs
                continue

            # If paragraph is very long, split by sentences
            if len(para) > 500:
                sentences = para.split('. ')
                current_chunk = ""

                for sentence in sentences:
                    if len(current_chunk + sentence) > 400:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence
                    else:
                        current_chunk += ". " + sentence if current_chunk else sentence

                if current_chunk:
                    chunks.append(current_chunk.strip())
            else:
                chunks.append(para)

        return chunks

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar content in the Wikipedia index"""
        if self.index.ntotal == 0:
            logger.warning("No documents in index. Add Wikipedia pages first.")
            return []

        # Generate query embedding
        query_embedding = self.model.encode([query], normalize_embeddings=True)

        # Search
        distances, indices = self.index.search(query_embedding, min(k, self.index.ntotal))

        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # Valid index
                results.append({
                    'content': self.sentences[idx],
                    'metadata': self.sentence_metadata[idx],
                    'distance': float(distances[0][i]),
                    'similarity_score': 1.0 - (float(distances[0][i]) / 2.0)  # Convert distance to similarity
                })

        return results

    def get_documents(self, query: str, k: int = 5) -> List[Document]:
        """Get LangChain Document objects from search results"""
        search_results = self.search(query, k)

        documents = []
        for result in search_results:
            doc = Document(
                page_content=result['content'],
                metadata={
                    **result['metadata'],
                    'similarity_score': result['similarity_score'],
                    'retrieval_method': 'wikipedia_faiss'
                }
            )
            documents.append(doc)

        return documents

    def build_index_from_topics(self, topics: List[str], max_pages_per_topic: int = 5):
        """Build index from a list of topics"""
        total_added = 0

        for topic in topics:
            logger.info(f"Processing topic: {topic}")

            # Search for pages related to topic
            try:
                search_results = wikipedia.search(topic, results=max_pages_per_topic)

                for page_title in search_results:
                    if self.add_wikipedia_page(page_title):
                        total_added += 1

            except Exception as e:
                logger.error(f"Failed to process topic '{topic}': {e}")

        # Save index after building
        if total_added > 0:
            self._save_index()
            logger.info(f"Built index with content from {total_added} pages")

        return total_added

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index"""
        return {
            'total_sentences': len(self.sentences),
            'total_pages': len(set(meta['page_title'] for meta in self.sentence_metadata)),
            'index_size_bytes': self.index.ntotal * self.model.get_sentence_embedding_dimension() * 4,  # float32
            'last_updated': max((meta['timestamp'] for meta in self.sentence_metadata), default="Never")
        }


class LegacyWikipediaRetriever:
    """Fallback retriever using LangChain's Wikipedia integration"""

    def __init__(self, cache_dir: str = "data/wikipedia_cache", max_cache_age_days: int = 7):
        self.cache_dir = cache_dir
        self.max_cache_age = timedelta(days=max_cache_age_days)
        os.makedirs(cache_dir, exist_ok=True)

        # Initialize LangChain Wikipedia retriever
        self.langchain_retriever = LangChainWikipediaRetriever(
            wiki_client=None,
            top_k_results=5,
            lang="en",
            load_all_available_meta=True,
            doc_content_chars_max=4000
        )

    def get_documents(self, query: str, k: int = 5) -> List[Document]:
        """Get documents using LangChain's Wikipedia retriever"""
        cache_key = self._get_cache_key(f"{query}_{k}")

        # Try cache first
        cached_docs = self._load_from_cache(cache_key)
        if cached_docs is not None:
            return cached_docs

        try:
            # Use LangChain retriever
            documents = self.langchain_retriever.get_relevant_documents(query)

            # Add metadata
            for doc in documents:
                doc.metadata['retrieval_method'] = 'langchain_wikipedia'
                doc.metadata['timestamp'] = datetime.now().isoformat()

            # Cache results
            self._save_to_cache(cache_key, documents)

            return documents[:k]

        except Exception as e:
            logger.error(f"LangChain Wikipedia retrieval failed: {e}")
            return []

    def _get_cache_key(self, query: str) -> str:
        return hashlib.md5(query.lower().encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> str:
        return os.path.join(self.cache_dir, f"legacy_{cache_key}.pkl")

    def _is_cache_valid(self, cache_path: str) -> bool:
        if not os.path.exists(cache_path):
            return False
        cache_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        return datetime.now() - cache_time < self.max_cache_age

    def _save_to_cache(self, cache_key: str, documents: List[Document]):
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(documents, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _load_from_cache(self, cache_key: str) -> Optional[List[Document]]:
        cache_path = self._get_cache_path(cache_key)
        if not self._is_cache_valid(cache_path):
            return None
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None
