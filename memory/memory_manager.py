import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from langchain.memory import VectorStoreRetrieverMemory
from langchain.schema import Document

from memory.memory_manager import EnhancedMemoryManager as BaseMemoryManager
from memory.vector_store import get_vector_store
from rag.composite_memory import CompositeRetriever  # Use existing class name
from rag.external_retriever import get_external_vector_store  # Use existing function
from prompts.prompt_enhancer import PromptEnhancer, auto_enhance_prompt, get_template_prompt
from prompts.prompt_enhancer import ResponseFormat, DetailLevel, PromptContext
import logging

logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_EXTERNAL_URLS = [
    "https://en.wikipedia.org/wiki/Meaning_of_life",
    "https://plato.stanford.edu/entries/consciousness/",
    "https://en.wikipedia.org/wiki/Philosophy_of_mind",
    "https://plato.stanford.edu/entries/artificial-intelligence/",
    "https://en.wikipedia.org/wiki/Cognitive_science"
]

class PromptAwareMemoryManager(BaseMemoryManager):
    """Enhanced memory manager with advanced prompt enhancement capabilities"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_enhancer = PromptEnhancer()
        self.enhancement_stats = {
            'total_enhancements': 0,
            'auto_enhancements': 0,
            'template_enhancements': 0,
            'manual_enhancements': 0
        }

    def get_enhanced_context_with_prompt(self,
                                         query: str,
                                         enhancement_type: str = "auto",
                                         response_format: ResponseFormat = ResponseFormat.DETAILED,
                                         detail_level: DetailLevel = DetailLevel.COMPREHENSIVE,
                                         context_info: dict = None,
                                         **kwargs) -> tuple:
        """
        Get both enhanced prompt and augmented context

        Returns:
            Tuple of (enhanced_prompt, augmented_context)
        """
        try:
            # Enhance the prompt
            if enhancement_type == "auto":
                enhanced_prompt = auto_enhance_prompt(query, **kwargs)
                self.enhancement_stats['auto_enhancements'] += 1
            elif enhancement_type == "template":
                template_type = kwargs.get('template_type')
                enhanced_prompt = get_template_prompt(query, template_type)
                self.enhancement_stats['template_enhancements'] += 1
            elif enhancement_type == "manual":
                prompt_context = PromptContext(**context_info) if context_info else None
                enhanced_prompt = self.prompt_enhancer.enhance_prompt(
                    query, response_format, detail_level, prompt_context,
                    kwargs.get('custom_instructions')
                )
                self.enhancement_stats['manual_enhancements'] += 1
            else:
                enhanced_prompt = query

            self.enhancement_stats['total_enhancements'] += 1

            # Get augmented context using the original query
            # (we search with original query but respond with enhanced prompt)
            augmented_context = self.get_augmented_context(
                query,
                include_conversation_history=kwargs.get('include_history', True),
                context_length_limit=kwargs.get('context_limit', 4000),
                k=kwargs.get('retrieval_k', self.retrieval_k)
            )

            logger.info(f"Enhanced prompt and retrieved context for query: '{query[:50]}...'")

            return enhanced_prompt, augmented_context

        except Exception as e:
            logger.error(f"Failed to enhance prompt and get context: {e}")
            return query, self.get_augmented_context(query)

    def create_full_prompt(self,
                           user_query: str,
                           system_context: str = None,
                           enhancement_config: dict = None) -> str:
        """
        Create a complete prompt with system context, enhanced query, and retrieved context
        """
        config = enhancement_config or {}

        # Get enhanced prompt and context
        enhanced_prompt, augmented_context = self.get_enhanced_context_with_prompt(
            user_query, **config
        )

        # Build complete prompt
        prompt_parts = []

        # System context
        if system_context:
            prompt_parts.append(f"SYSTEM CONTEXT:\n{system_context}")

        # Retrieved context
        if augmented_context:
            prompt_parts.append(f"RETRIEVED CONTEXT:\n{augmented_context}")

        # Enhanced user prompt
        prompt_parts.append(f"USER REQUEST:\n{enhanced_prompt}")

        # Final instructions
        prompt_parts.append("""
RESPONSE INSTRUCTIONS:
- Use the retrieved context to inform your response but don't limit yourself to it
- Follow the structured format specified in the user request
- Ensure your response is comprehensive and well-organized
- Cite sources from the context when relevant
- If the context doesn't fully address the query, acknowledge this and provide the best response possible
""")

        full_prompt = "\n\n" + "=" * 50 + "\n\n".join(prompt_parts)

        logger.info(f"Created full prompt of {len(full_prompt)} characters")
        return full_prompt

    def get_enhancement_stats(self):
        """Get prompt enhancement statistics"""
        return {
            **super().get_memory_stats(),
            'prompt_enhancement_stats': self.enhancement_stats
        }


class EnhancedMemoryManager:
    """
    Enhanced memory manager that provides:
    1. Persistent conversation memory
    2. Multi-source knowledge retrieval (internal, external, Wikipedia)
    3. Context management and optimization
    4. Memory statistics and health monitoring
    """

    def __init__(self,
                 external_urls: Optional[List[str]] = None,
                 memory_k: int = 10,
                 retrieval_k: int = 6):

        self.memory_k = memory_k
        self.retrieval_k = retrieval_k
        self.conversation_history = []
        self.external_urls = external_urls or DEFAULT_EXTERNAL_URLS

        # Initialize internal vector store and memory
        self._setup_internal_memory()

        # Setup composite retriever with multiple sources
        self._setup_composite_retriever()

        # Statistics tracking
        self.stats = {
            'total_retrievals': 0,
            'successful_saves': 0,
            'failed_saves': 0,
            'last_retrieval': None,
            'last_save': None
        }

        logger.info("Enhanced Memory Manager initialized successfully")

    def _setup_internal_memory(self):
        """Setup internal vector store and retriever memory"""
        try:
            self.vector_store = get_vector_store()
            self.retriever_memory = VectorStoreRetrieverMemory(
                retriever=self.vector_store.as_retriever(search_kwargs={"k": self.memory_k})
            )
            logger.info("Internal memory system initialized")
        except Exception as e:
            logger.error(f"Failed to initialize internal memory: {e}")
            raise

    def _setup_composite_retriever(self):
        """Setup composite retriever with all knowledge sources"""
        try:
            # Create external vector store using existing function
            external_vector_store = None
            if self.external_urls:
                try:
                    external_vector_store = get_external_vector_store(self.external_urls)
                    logger.info(f"External vector store initialized with {len(self.external_urls)} URLs")
                except Exception as e:
                    logger.warning(f"Failed to initialize external vector store: {e}")
                    external_vector_store = None

            # Create composite retriever using existing class
            self.composite_retriever = CompositeRetriever(
                internal_retriever=self.vector_store.as_retriever(),
                external_vector_store=external_vector_store
            )

            logger.info("Composite retriever initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize composite retriever: {e}")
            # Fallback to internal retriever only
            self.composite_retriever = None

    def get_augmented_context(self,
                              query: str,
                              include_conversation_history: bool = True,
                              context_length_limit: int = 4000,
                              k: int = None) -> str:
        """
        Get augmented context combining multiple knowledge sources
        
        Args:
            query: The query to search for
            include_conversation_history: Whether to include recent conversation history
            context_length_limit: Maximum characters in returned context
            k: Number of documents to retrieve (uses self.retrieval_k if None)
        
        Returns:
            Combined context string
        """
        try:
            self.stats['total_retrievals'] += 1
            self.stats['last_retrieval'] = datetime.now().isoformat()

            k = k or self.retrieval_k
            context_parts = []

            # Get conversation history context
            if include_conversation_history:
                history_context = self._get_conversation_context(query)
                if history_context:
                    context_parts.append("=== CONVERSATION HISTORY ===")
                    context_parts.append(history_context)

            # Get knowledge base context
            if self.composite_retriever:
                try:
                    knowledge_context = self.composite_retriever.get_combined_context(query, k=k)
                    if knowledge_context:
                        context_parts.append("=== KNOWLEDGE BASE ===")
                        context_parts.append(knowledge_context)
                except Exception as e:
                    logger.warning(f"Composite retriever failed: {e}")
                    # Fallback to basic vector store search
                    fallback_context = self._get_fallback_context(query, k=k)
                    if fallback_context:
                        context_parts.append("=== KNOWLEDGE BASE (FALLBACK) ===")
                        context_parts.append(fallback_context)

            # Combine and limit context length
            full_context = "\n\n".join(context_parts)

            if len(full_context) > context_length_limit:
                # Truncate while preserving structure
                truncated_context = full_context[:context_length_limit]
                # Try to cut at a section boundary
                last_section = truncated_context.rfind("===")
                if last_section > context_length_limit * 0.7:  # If we can preserve most content
                    truncated_context = truncated_context[:last_section]

                full_context = truncated_context + "\n\n[... context truncated ...]"

            logger.info(f"Retrieved augmented context for query: '{query[:50]}...'")
            return full_context

        except Exception as e:
            logger.error(f"Failed to get augmented context: {e}")
            # Fallback to basic retrieval
            return self._get_fallback_context(query, k=k or 3)

    def _get_conversation_context(self, query: str) -> str:
        """Get relevant conversation history"""
        try:
            # Use retriever memory to get relevant past conversations
            relevant_memory = self.retriever_memory.load_memory_variables({"query": query})
            history = relevant_memory.get("history", "")

            # Also include recent conversation history
            recent_history = self._get_recent_history(limit=3)

            combined_history = []
            if history:
                combined_history.append("Relevant past conversations:")
                combined_history.append(history)

            if recent_history:
                combined_history.append("Recent conversation:")
                combined_history.append(recent_history)

            return "\n\n".join(combined_history) if combined_history else ""

        except Exception as e:
            logger.warning(f"Failed to get conversation context: {e}")
            return ""

    def _get_recent_history(self, limit: int = 3) -> str:
        """Get recent conversation history"""
        if not self.conversation_history:
            return ""

        recent = self.conversation_history[-limit:]
        history_parts = []

        for entry in recent:
            timestamp = entry.get('timestamp', 'Unknown time')
            input_text = entry.get('input', '')[:200]  # Limit length
            output_text = entry.get('output', '')[:200]  # Limit length

            history_parts.append(f"[{timestamp}]")
            history_parts.append(f"User: {input_text}")
            history_parts.append(f"Assistant: {output_text}")
            history_parts.append("---")

        return "\n".join(history_parts[:-1]) if history_parts else ""  # Remove last separator

    def _get_fallback_context(self, query: str, k: int = 3) -> str:
        """Fallback context retrieval when main system fails"""
        try:
            # Try basic vector store retrieval
            docs = self.vector_store.similarity_search(query, k=k)
            if docs:
                return "\n\n".join(doc.page_content for doc in docs)
            else:
                return "No relevant context found."
        except Exception as e:
            logger.error(f"Fallback context retrieval failed: {e}")
            return "Context retrieval unavailable."

    def save_interaction(self,
                         user_input: str,
                         assistant_output: str,
                         metadata: Optional[Dict[str, Any]] = None):
        """
        Save interaction to both retriever memory and conversation history
        
        Args:
            user_input: User's input/query
            assistant_output: Assistant's response
            metadata: Optional metadata about the interaction
        """
        try:
            # Save to retriever memory for semantic search
            self.retriever_memory.save_context(
                {"input": user_input},
                {"output": assistant_output}
            )

            # Save to conversation history with timestamp
            interaction = {
                'timestamp': datetime.now().isoformat(),
                'input': user_input,
                'output': assistant_output,
                'metadata': metadata or {}
            }

            self.conversation_history.append(interaction)

            # Keep conversation history manageable
            if len(self.conversation_history) > 100:
                self.conversation_history = self.conversation_history[-50:]

            self.stats['successful_saves'] += 1
            self.stats['last_save'] = datetime.now().isoformat()

            logger.info("Interaction saved successfully")

        except Exception as e:
            logger.error(f"Failed to save interaction: {e}")
            self.stats['failed_saves'] += 1

    def add_knowledge_document(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a new document to the internal knowledge base"""
        try:
            doc = Document(
                page_content=content,
                metadata=metadata or {'source': 'manual_addition', 'timestamp': datetime.now().isoformat()}
            )

            # Add to vector store
            self.vector_store.add_documents([doc])

            logger.info("Knowledge document added successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to add knowledge document: {e}")
            return False

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory system statistics"""
        try:
            stats = {
                **self.stats,
                'conversation_history_length': len(self.conversation_history),
                'internal_memory_available': self.retriever_memory is not None,
                'composite_retriever_available': self.composite_retriever is not None,
                'external_urls_configured': len(self.external_urls) if self.external_urls else 0
            }

            return stats

        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {'error': str(e)}

    def clear_conversation_history(self):
        """Clear conversation history but keep knowledge base"""
        self.conversation_history.clear()
        logger.info("Conversation history cleared")

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on memory system components"""
        health = {
            'overall_status': 'healthy',
            'components': {},
            'issues': []
        }

        try:
            # Check internal vector store
            try:
                test_docs = self.vector_store.similarity_search("test", k=1)
                health['components']['internal_vector_store'] = 'healthy'
            except Exception as e:
                health['components']['internal_vector_store'] = 'unhealthy'
                health['issues'].append(f"Internal vector store: {str(e)}")

            # Check retriever memory
            try:
                test_memory = self.retriever_memory.load_memory_variables({"test": "test"})
                health['components']['retriever_memory'] = 'healthy'
            except Exception as e:
                health['components']['retriever_memory'] = 'unhealthy'
                health['issues'].append(f"Retriever memory: {str(e)}")

            # Check composite retriever
            if self.composite_retriever:
                try:
                    test_context = self.composite_retriever.get_combined_context("test", k=1)
                    health['components']['composite_retriever'] = 'healthy'
                except Exception as e:
                    health['components']['composite_retriever'] = 'unhealthy'
                    health['issues'].append(f"Composite retriever: {str(e)}")
            else:
                health['components']['composite_retriever'] = 'not_available'

            # Overall status
            if health['issues']:
                health['overall_status'] = 'degraded' if len(health['issues']) < 2 else 'unhealthy'

            return health

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'overall_status': 'error',
                'error': str(e)
            }


# Original MemoryManager class for backward compatibility
class MemoryManager:
    """Original MemoryManager class with enhanced functionality"""

    def __init__(self, external_urls: Optional[List[str]] = None):
        # Use the existing pattern from the original code
        self.vector_store = get_vector_store()
        self.retriever = VectorStoreRetrieverMemory(retriever=self.vector_store.as_retriever())

        # Setup external vector store
        urls = external_urls or DEFAULT_EXTERNAL_URLS
        try:
            self.external_vector_store = get_external_vector_store(urls)
        except Exception as e:
            logger.warning(f"Failed to initialize external vector store: {e}")
            self.external_vector_store = None

        # Setup composite retriever
        try:
            self.composite_retriever = CompositeRetriever(
                internal_retriever=self.vector_store.as_retriever(),
                external_vector_store=self.external_vector_store
            )
        except Exception as e:
            logger.error(f"Failed to initialize composite retriever: {e}")
            self.composite_retriever = None

        # Enhanced functionality
        self.conversation_history = []
        self.stats = {'total_retrievals': 0, 'successful_saves': 0, 'failed_saves': 0}

    def get_augmented_context(self, query, k=4):
        """Get augmented context using composite retriever"""
        try:
            self.stats['total_retrievals'] += 1

            if self.composite_retriever:
                context = self.composite_retriever.get_combined_context(query, k=k)
                logger.info(f"Retrieved augmented context for query: '{query[:50]}...'")
                return context
            else:
                # Fallback to basic vector store
                docs = self.vector_store.similarity_search(query, k=k)
                context = "\n\n".join(doc.page_content for doc in docs)
                logger.warning("Used fallback context retrieval")
                return context

        except Exception as e:
            logger.error(f"Failed to get augmented context: {e}")
            return "Context retrieval failed."

    def save(self, context, output):
        """Save interaction to memory"""
        try:
            self.retriever.save_context({"input": context}, {"output": output})

            # Also save to conversation history
            self.conversation_history.append({
                'timestamp': datetime.now().isoformat(),
                'input': context,
                'output': output
            })

            # Keep history manageable
            if len(self.conversation_history) > 50:
                self.conversation_history = self.conversation_history[-25:]

            self.stats['successful_saves'] += 1
            logger.info("Interaction saved successfully")

        except Exception as e:
            logger.error(f"Failed to save interaction: {e}")
            self.stats['failed_saves'] += 1

    @property
    def retriever_memory(self):
        """Property for backward compatibility"""
        return self.retriever

    def get_stats(self):
        """Get basic statistics"""
        return {
            **self.stats,
            'conversation_history_length': len(self.conversation_history),
            'composite_retriever_available': self.composite_retriever is not None,
            'external_vector_store_available': self.external_vector_store is not None
        }


# Factory function for easy setup
def create_memory_manager(enhanced: bool = False,
                          external_urls: Optional[List[str]] = None,
                          **kwargs) -> MemoryManager:
    """
    Create a memory manager with optional enhancement
    
    Args:
        enhanced: Whether to use EnhancedMemoryManager
        external_urls: List of URLs for external knowledge
        **kwargs: Additional arguments for EnhancedMemoryManager
    
    Returns:
        MemoryManager or EnhancedMemoryManager instance
    """
    if enhanced:
        return EnhancedMemoryManager(external_urls=external_urls, **kwargs)
    else:
        return MemoryManager(external_urls=external_urls)


# Usage example
if __name__ == "__main__":
    # Test with original interface
    print("Testing original MemoryManager interface...")

    try:
        memory_manager = MemoryManager()

        # Test retrieval
        query = "What is artificial intelligence?"
        context = memory_manager.get_augmented_context(query)

        print("Augmented Context:")
        print("=" * 50)
        print(context[:500] + "..." if len(context) > 500 else context)

        # Test saving
        response = "AI is the simulation of human intelligence in machines."
        memory_manager.save(query, response)

        # Get stats
        print("\n" + "=" * 50)
        print("Memory Stats:")
        stats = memory_manager.get_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")

        print("\nOriginal MemoryManager test completed successfully!")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"Test failed: {e}")
