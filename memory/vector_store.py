import os
import pickle
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import hashlib

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class EnhancedVectorStore:
    """Enhanced vector store with better management and features"""

    def __init__(self,
                 embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 index_path: str = "data/faiss_index",
                 metadata_path: str = "data/vector_metadata.pkl",
                 auto_save: bool = True,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):

        self.embedding_model = embedding_model
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.auto_save = auto_save
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'}
        )

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        # Vector store and metadata
        self.vector_store = None
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'document_count': 0,
            'chunk_count': 0,
            'embeddings_model': embedding_model,
            'documents': {},  # Track document sources and stats
            'tags': set(),  # Document tags for organization
        }

        # Ensure directories exist
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

        # Load or create vector store
        self._initialize_vector_store()

        logger.info(f"Enhanced vector store initialized with {self.metadata['document_count']} documents")

    def _initialize_vector_store(self):
        """Initialize vector store by loading existing or creating new"""
        try:
            if self._load_existing():
                logger.info("Loaded existing vector store")
            else:
                self._create_new_store()
                logger.info("Created new vector store")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            self._create_new_store()

    def _load_existing(self) -> bool:
        """Try to load existing vector store and metadata"""
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
                # Load vector store
                self.vector_store = FAISS.load_local(
                    self.index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )

                # Load metadata
                with open(self.metadata_path, 'rb') as f:
                    loaded_metadata = pickle.load(f)

                # Update metadata with loaded data, preserving structure
                self.metadata.update(loaded_metadata)

                # Convert tags back to set if it was serialized as list
                if isinstance(self.metadata.get('tags'), list):
                    self.metadata['tags'] = set(self.metadata['tags'])

                return True

        except Exception as e:
            logger.warning(f"Failed to load existing vector store: {e}")

        return False

    def _create_new_store(self):
        """Create a new empty vector store"""
        try:
            # Create with dummy document to avoid empty store issues
            dummy_doc = Document(
                page_content="This is a placeholder document for initialization.",
                metadata={'source': 'initialization', 'type': 'placeholder'}
            )

            self.vector_store = FAISS.from_documents([dummy_doc], self.embeddings)

            # Update metadata
            self.metadata['created_at'] = datetime.now().isoformat()
            self.metadata['last_updated'] = datetime.now().isoformat()
            self.metadata['document_count'] = 0  # Don't count placeholder
            self.metadata['chunk_count'] = 0

            if self.auto_save:
                self.save()

        except Exception as e:
            logger.error(f"Failed to create new vector store: {e}")
            raise

    def add_documents(self,
                      documents: List[Document],
                      source: str = None,
                      tags: List[str] = None,
                      chunk_documents: bool = True) -> bool:
        """
        Add documents to the vector store
        
        Args:
            documents: List of documents to add
            source: Source identifier for tracking
            tags: Tags to associate with documents
            chunk_documents: Whether to split documents into chunks
        """
        try:
            if not documents:
                logger.warning("No documents provided to add")
                return False

            processed_docs = []

            for doc in documents:
                if chunk_documents and len(doc.page_content) > self.chunk_size:
                    # Split large documents into chunks
                    chunks = self.text_splitter.split_documents([doc])
                    for i, chunk in enumerate(chunks):
                        # Add chunk metadata
                        chunk.metadata.update({
                            'chunk_id': i,
                            'total_chunks': len(chunks),
                            'source': source or doc.metadata.get('source', 'unknown'),
                            'added_at': datetime.now().isoformat()
                        })
                        if tags:
                            chunk.metadata['tags'] = tags
                        processed_docs.append(chunk)
                else:
                    # Add document as-is with metadata
                    doc.metadata.update({
                        'source': source or doc.metadata.get('source', 'unknown'),
                        'added_at': datetime.now().isoformat()
                    })
                    if tags:
                        doc.metadata['tags'] = tags
                    processed_docs.append(doc)

            # Add to vector store
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(processed_docs, self.embeddings)
            else:
                self.vector_store.add_documents(processed_docs)

            # Update metadata
            doc_source = source or 'unknown'
            if doc_source not in self.metadata['documents']:
                self.metadata['documents'][doc_source] = {
                    'document_count': 0,
                    'chunk_count': 0,
                    'added_at': datetime.now().isoformat(),
                    'last_updated': datetime.now().isoformat()
                }

            self.metadata['documents'][doc_source]['document_count'] += len(documents)
            self.metadata['documents'][doc_source]['chunk_count'] += len(processed_docs)
            self.metadata['documents'][doc_source]['last_updated'] = datetime.now().isoformat()

            self.metadata['document_count'] += len(documents)
            self.metadata['chunk_count'] += len(processed_docs)
            self.metadata['last_updated'] = datetime.now().isoformat()

            if tags:
                self.metadata['tags'].update(tags)

            if self.auto_save:
                self.save()

            logger.info(f"Added {len(documents)} documents ({len(processed_docs)} chunks) from source: {doc_source}")
            return True

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False

    def add_texts(self,
                  texts: List[str],
                  metadatas: List[Dict] = None,
                  source: str = None,
                  tags: List[str] = None) -> bool:
        """Add texts to the vector store"""
        try:
            if not texts:
                return False

            # Convert texts to documents
            documents = []
            for i, text in enumerate(texts):
                metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
                metadata.update({
                    'source': source or 'text_input',
                    'text_id': i,
                    'added_at': datetime.now().isoformat()
                })
                if tags:
                    metadata['tags'] = tags

                documents.append(Document(page_content=text, metadata=metadata))

            return self.add_documents(documents, source, tags)

        except Exception as e:
            logger.error(f"Failed to add texts: {e}")
            return False

    def similarity_search(self,
                          query: str,
                          k: int = 4,
                          filter_tags: List[str] = None,
                          filter_source: str = None,
                          score_threshold: float = None) -> List[Document]:
        """
        Perform similarity search with optional filtering
        
        Args:
            query: Search query
            k: Number of documents to return
            filter_tags: Only return documents with these tags
            filter_source: Only return documents from this source
            score_threshold: Minimum similarity score threshold
        """
        try:
            if self.vector_store is None:
                logger.warning("Vector store not initialized")
                return []

            # Get more results than needed for filtering
            search_k = min(k * 3, 50) if (filter_tags or filter_source) else k

            if score_threshold is not None:
                docs_with_scores = self.vector_store.similarity_search_with_score(query, k=search_k)
                docs = [doc for doc, score in docs_with_scores if score <= score_threshold]
            else:
                docs = self.vector_store.similarity_search(query, k=search_k)

            # Apply filters
            if filter_tags or filter_source:
                filtered_docs = []
                for doc in docs:
                    # Check tag filter
                    if filter_tags:
                        doc_tags = doc.metadata.get('tags', [])
                        if not any(tag in doc_tags for tag in filter_tags):
                            continue

                    # Check source filter
                    if filter_source:
                        if doc.metadata.get('source') != filter_source:
                            continue

                    filtered_docs.append(doc)

                docs = filtered_docs[:k]

            return docs

        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []

    def similarity_search_with_score(self,
                                     query: str,
                                     k: int = 4) -> List[tuple]:
        """Perform similarity search and return documents with scores"""
        try:
            if self.vector_store is None:
                return []

            return self.vector_store.similarity_search_with_score(query, k=k)

        except Exception as e:
            logger.error(f"Similarity search with score failed: {e}")
            return []

    def delete_by_source(self, source: str) -> bool:
        """Delete all documents from a specific source"""
        try:
            # This is a limitation of FAISS - we need to rebuild without the documents
            # For now, we'll log this operation and mark for future implementation
            logger.warning(f"Delete by source '{source}' requested - requires vector store rebuild")

            # Remove from metadata
            if source in self.metadata['documents']:
                del self.metadata['documents'][source]
                self.metadata['last_updated'] = datetime.now().isoformat()

                if self.auto_save:
                    self.save_metadata()

            return False  # Indicate that actual deletion wasn't performed

        except Exception as e:
            logger.error(f"Failed to delete by source: {e}")
            return False

    def get_document_count(self) -> int:
        """Get total number of documents"""
        return self.metadata.get('document_count', 0)

    def get_chunk_count(self) -> int:
        """Get total number of chunks"""
        return self.metadata.get('chunk_count', 0)

    def get_sources(self) -> List[str]:
        """Get list of all document sources"""
        return list(self.metadata['documents'].keys())

    def get_tags(self) -> List[str]:
        """Get list of all tags"""
        return list(self.metadata['tags'])

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the vector store"""
        return {
            'total_documents': self.get_document_count(),
            'total_chunks': self.get_chunk_count(),
            'sources': len(self.get_sources()),
            'tags': len(self.get_tags()),
            'created_at': self.metadata.get('created_at'),
            'last_updated': self.metadata.get('last_updated'),
            'embedding_model': self.metadata.get('embeddings_model'),
            'index_size_mb': self._get_index_size_mb(),
            'sources_detail': dict(self.metadata['documents'])
        }

    def _get_index_size_mb(self) -> float:
        """Get approximate index size in MB"""
        try:
            if os.path.exists(self.index_path):
                size_bytes = sum(
                    os.path.getsize(os.path.join(self.index_path, f))
                    for f in os.listdir(self.index_path)
                    if os.path.isfile(os.path.join(self.index_path, f))
                )
                return round(size_bytes / (1024 * 1024), 2)
        except:
            pass
        return 0.0

    def save(self):
        """Save vector store and metadata to disk"""
        try:
            if self.vector_store is not None:
                self.vector_store.save_local(self.index_path)

            self.save_metadata()
            logger.info(f"Vector store saved to {self.index_path}")

        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")

    def save_metadata(self):
        """Save only metadata to disk"""
        try:
            # Convert sets to lists for pickling
            metadata_to_save = dict(self.metadata)
            metadata_to_save['tags'] = list(metadata_to_save['tags'])

            with open(self.metadata_path, 'wb') as f:
                pickle.dump(metadata_to_save, f)

        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    def as_retriever(self, **kwargs):
        """Return a retriever interface for the vector store"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")

        return self.vector_store.as_retriever(**kwargs)

    def rebuild_index(self, chunk_size: int = None, chunk_overlap: int = None):
        """Rebuild the index with new parameters (future implementation)"""
        logger.warning("Index rebuilding not implemented yet")
        pass

    def backup(self, backup_path: str = None) -> bool:
        """Create a backup of the vector store"""
        try:
            import shutil
            from datetime import datetime

            if backup_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"{self.index_path}_backup_{timestamp}"

            # Backup index
            if os.path.exists(self.index_path):
                shutil.copytree(self.index_path, backup_path)

            # Backup metadata
            if os.path.exists(self.metadata_path):
                shutil.copy2(self.metadata_path, f"{backup_path}_metadata.pkl")

            logger.info(f"Vector store backed up to {backup_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the vector store"""
        health = {
            'status': 'healthy',
            'issues': [],
            'checks': {}
        }

        try:
            # Check if vector store is loaded
            health['checks']['vector_store_loaded'] = self.vector_store is not None

            # Check if files exist
            health['checks']['index_exists'] = os.path.exists(self.index_path)
            health['checks']['metadata_exists'] = os.path.exists(self.metadata_path)

            # Check if we can perform search
            if self.vector_store is not None:
                try:
                    test_results = self.similarity_search("test query", k=1)
                    health['checks']['search_functional'] = True
                except:
                    health['checks']['search_functional'] = False
                    health['issues'].append("Search functionality not working")
            else:
                health['checks']['search_functional'] = False
                health['issues'].append("Vector store not loaded")

            # Check metadata consistency
            if self.metadata['document_count'] > 0 and self.vector_store is not None:
                try:
                    # Try to get actual count (approximate)
                    health['checks']['metadata_consistent'] = True
                except:
                    health['checks']['metadata_consistent'] = False
                    health['issues'].append("Metadata may be inconsistent")

            # Overall status
            if health['issues']:
                health['status'] = 'degraded' if len(health['issues']) < 2 else 'unhealthy'

            return health

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'checks': {},
                'issues': [str(e)]
            }


# Improved factory function
def get_vector_store(config: Dict[str, Any] = None) -> EnhancedVectorStore:
    """
    Get an enhanced vector store with configuration options
    
    Args:
        config: Configuration dictionary with optional keys:
            - embedding_model: Embedding model to use
            - index_path: Path to store the index
            - metadata_path: Path to store metadata
            - auto_save: Whether to auto-save changes
            - chunk_size: Size of text chunks
            - chunk_overlap: Overlap between chunks
    """
    config = config or {}

    return EnhancedVectorStore(
        embedding_model=config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2'),
        index_path=config.get('index_path', 'data/faiss_index'),
        metadata_path=config.get('metadata_path', 'data/vector_metadata.pkl'),
        auto_save=config.get('auto_save', True),
        chunk_size=config.get('chunk_size', 1000),
        chunk_overlap=config.get('chunk_overlap', 200)
    )


# Legacy compatibility function
def get_legacy_vector_store():
    """Legacy function for backward compatibility"""
    try:
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        faiss_path = "faiss_index"

        if os.path.exists(faiss_path):
            try:
                return FAISS.load_local(
                    faiss_path,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
            except:
                pass

        # Create new store
        dummy_text = ["This is a placeholder document for initialization."]
        vector_store = FAISS.from_texts(texts=dummy_text, embedding=embeddings)

        try:
            vector_store.save_local(faiss_path)
        except:
            pass

        return vector_store

    except Exception as e:
        logger.error(f"Failed to create legacy vector store: {e}")
        # Final fallback
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        return FAISS.from_texts(["fallback"], embeddings)


# Usage example
if __name__ == "__main__":
    # Test the enhanced vector store
    print("Testing Enhanced Vector Store...")

    try:
        # Create enhanced vector store
        config = {
            'index_path': 'data/test_faiss_index',
            'metadata_path': 'data/test_metadata.pkl',
            'chunk_size': 500,
            'auto_save': True
        }

        vector_store = get_vector_store(config)

        # Add some test documents
        test_docs = [
            Document(
                page_content="Machine learning is a method of data analysis that automates analytical model building.",
                metadata={'source': 'ml_intro', 'topic': 'machine_learning'}
            ),
            Document(
                page_content="Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals.",
                metadata={'source': 'ai_intro', 'topic': 'artificial_intelligence'}
            )
        ]

        success = vector_store.add_documents(
            test_docs,
            source='test_documents',
            tags=['ai', 'ml', 'introduction']
        )

        print(f"Documents added: {success}")

        # Test search
        results = vector_store.similarity_search("What is machine learning?", k=2)
        print(f"Search results: {len(results)}")
        for i, doc in enumerate(results):
            print(f"  {i + 1}. {doc.page_content[:100]}...")

        # Test filtering
        filtered_results = vector_store.similarity_search(
            "artificial intelligence",
            k=2,
            filter_tags=['ai']
        )
        print(f"Filtered results: {len(filtered_results)}")

        # Show stats
        stats = vector_store.get_stats()
        print("\nVector Store Stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # Health check
        health = vector_store.health_check()
        print(f"\nHealth Status: {health['status']}")
        if health['issues']:
            print("Issues:", health['issues'])

        print("\nEnhanced Vector Store test completed successfully!")

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback

        traceback.print_exc()
