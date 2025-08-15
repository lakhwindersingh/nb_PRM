# memory/memory_manager.py - Complete fix for MemoryManager class

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from langchain.memory import VectorStoreRetrieverMemory
from langchain.schema import Document

from memory.vector_store import get_vector_store
from rag.composite_memory import CompositeRetriever
from rag.external_retriever import get_external_vector_store

logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_EXTERNAL_URLS = [
    "https://en.wikipedia.org/wiki/Meaning_of_life",
    "https://plato.stanford.edu/entries/consciousness/",
    "https://en.wikipedia.org/wiki/Philosophy_of_mind",
    "https://plato.stanford.edu/entries/artificial-intelligence/",
    "https://en.wikipedia.org/wiki/Cognitive_science"
]


class ResponseStyle:
    """Simple response styles for compatibility"""
    DETAILED = "detailed"
    ANALYTICAL = "analytical"
    HOWTO = "howto"
    COMPARISON = "comparison"
    CREATIVE = "creative"
    RESEARCH = "research"
    CODING = "coding"
    EXPLANATION = "explanation"
    AUTO = "auto"


class PromptEnhancer:
    """Built-in prompt enhancement system"""

    @staticmethod
    def enhance_prompt(prompt: str, style: str = "detailed") -> str:
        """Enhance a prompt based on the specified style"""

        if style == "auto":
            style = PromptEnhancer._detect_style(prompt)

        enhancers = {
            "detailed": PromptEnhancer._enhance_detailed,
            "analytical": PromptEnhancer._enhance_analytical,
            "howto": PromptEnhancer._enhance_howto,
            "comparison": PromptEnhancer._enhance_comparison,
            "creative": PromptEnhancer._enhance_creative,
            "research": PromptEnhancer._enhance_research,
            "coding": PromptEnhancer._enhance_coding,
            "explanation": PromptEnhancer._enhance_explanation
        }

        enhancer = enhancers.get(style, enhancers["detailed"])
        return enhancer(prompt)

    @staticmethod
    def _detect_style(prompt: str) -> str:
        """Auto-detect the best enhancement style for a prompt"""
        prompt_lower = prompt.lower()

        if any(word in prompt_lower for word in ['how to', 'steps', 'process', 'guide', 'tutorial']):
            return "howto"
        elif any(word in prompt_lower for word in ['compare', 'vs', 'versus', 'difference', 'better']):
            return "comparison"
        elif any(word in prompt_lower for word in ['analyze', 'analysis', 'evaluate', 'assess', 'examine']):
            return "analytical"
        elif any(word in prompt_lower for word in ['create', 'design', 'write', 'compose', 'generate']):
            return "creative"
        elif any(word in prompt_lower for word in ['research', 'study', 'investigate', 'find out', 'sources']):
            return "research"
        elif any(word in prompt_lower for word in ['code', 'program', 'implement', 'algorithm', 'function']):
            return "coding"
        elif any(word in prompt_lower for word in ['explain', 'what is', 'definition', 'meaning', 'understand']):
            return "explanation"
        else:
            return "detailed"

    @staticmethod
    def _enhance_detailed(prompt: str) -> str:
        return f"""
You are a knowledgeable expert providing comprehensive information. Structure your response as follows:

**Query:** {prompt}

**Response Requirements:**
1. **Executive Summary** - Key points and main conclusions upfront
2. **Detailed Analysis** - Thorough exploration with multiple perspectives
3. **Supporting Evidence** - Facts, data, examples, and sources
4. **Practical Applications** - Real-world relevance and use cases
5. **Implications & Context** - Broader significance and connections
6. **Conclusion** - Summary of key takeaways

Make your response thorough, well-organized, and valuable for someone seeking deep understanding of this topic.
"""

    @staticmethod
    def _enhance_analytical(prompt: str) -> str:
        return f"""
You are an expert analyst conducting a thorough examination. Provide a structured analytical response:

**Analysis Subject:** {prompt}

**Analytical Framework:**
1. **Problem Definition** - Clearly define what is being analyzed
2. **Methodology** - Explain your analytical approach and criteria
3. **Data & Evidence** - Present relevant information and sources
4. **Critical Analysis** - Systematic examination with reasoning
5. **Key Findings** - Main discoveries and insights
6. **Conclusions & Recommendations** - Evidence-based outcomes

Deliver an objective, thorough analysis that demonstrates critical thinking and provides valuable insights.
"""

    @staticmethod
    def _enhance_howto(prompt: str) -> str:
        return f"""
You are an expert instructor creating a comprehensive guide. Provide clear, actionable instructions:

**Instruction Request:** {prompt}

**Guide Structure:**
1. **Overview** - What will be accomplished and expected outcomes
2. **Prerequisites** - Required knowledge, skills, tools, or materials
3. **Step-by-Step Instructions** - Detailed, numbered process
4. **Tips & Best Practices** - Professional insights and optimizations
5. **Common Pitfalls** - What to avoid and troubleshooting
6. **Verification** - How to confirm success and quality checks

Create instructions that someone can follow successfully, even without prior experience.
"""

    @staticmethod
    def _enhance_comparison(prompt: str) -> str:
        return f"""
You are a comparison expert providing comprehensive evaluation. Structure your analysis:

**Comparison Topic:** {prompt}

**Comparison Framework:**
1. **Comparison Overview** - What's being compared and why
2. **Evaluation Criteria** - Key factors and metrics for comparison
3. **Feature Comparison** - Side-by-side analysis of key attributes
4. **Strengths & Weaknesses** - Pros and cons for each option
5. **Final Verdict** - Recommendations with clear reasoning

Deliver a balanced comparison that helps readers make informed decisions.
"""

    @staticmethod
    def _enhance_creative(prompt: str) -> str:
        return f"""
You are a creative expert and innovative thinker. Approach this with imagination and originality:

**Creative Challenge:** {prompt}

**Creative Structure:**
1. **Creative Vision** - Innovative concept and inspiration
2. **Creative Development** - Detailed exploration of possibilities
3. **Multiple Approaches** - Different creative directions and styles
4. **Concrete Examples** - Specific illustrations and demonstrations
5. **Implementation Ideas** - How to bring concepts to life

Be imaginative, original, and inspiring while providing practical guidance.
"""

    @staticmethod
    def _enhance_research(prompt: str) -> str:
        return f"""
You are a research expert providing comprehensive investigation results:

**Research Question:** {prompt}

**Research Report Structure:**
1. **Research Objective** - What question is being investigated
2. **Background Context** - Relevant foundational information
3. **Key Findings** - Main discoveries and data points
4. **Supporting Evidence** - Sources, studies, and documentation
5. **Research Implications** - Significance and applications

Provide research-quality information that would be valuable for academic or professional purposes.
"""

    @staticmethod
    def _enhance_coding(prompt: str) -> str:
        return f"""
You are an expert software engineer. Provide comprehensive coding guidance:

**Programming Request:** {prompt}

**Development Structure:**
1. **Problem Analysis** - Break down requirements and constraints
2. **Solution Design** - Architecture, approach, and technology choices
3. **Implementation** - Complete, working code with clear comments
4. **Code Walkthrough** - Explanation of key logic and components
5. **Testing & Validation** - Test cases and quality assurance
6. **Documentation** - Usage examples and maintenance notes

Deliver production-quality code with thorough explanation.
"""

    @staticmethod
    def _enhance_explanation(prompt: str) -> str:
        return f"""
You are an expert educator providing clear, comprehensive explanations:

**Topic to Explain:** {prompt}

**Educational Structure:**
1. **Introduction** - Overview of what will be learned
2. **Key Concepts** - Essential definitions and foundational ideas
3. **Detailed Explanation** - Step-by-step breakdown of the topic
4. **Examples & Illustrations** - Concrete demonstrations and analogies
5. **Practical Applications** - Real-world uses and relevance
6. **Summary** - Key takeaways

Create an explanation that would help someone truly understand the topic.
"""


class EnhancedMemoryManager:
    """Enhanced memory manager with all necessary features"""

    def __init__(self,
                 external_urls: Optional[List[str]] = None,
                 memory_k: int = 10,
                 retrieval_k: int = 6):

        self.memory_k = memory_k
        self.retrieval_k = retrieval_k
        self.conversation_history = []  # Initialize conversation history
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
            'last_save': None,
            'enhancement_stats': {
                'total_enhancements': 0,
                'by_style': {}
            }
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
        """Get augmented context combining multiple knowledge sources"""
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

            # Combine contexts
            full_context = "\n\n".join(context_parts)

            # Apply length limit
            if len(full_context) > context_length_limit:
                # Truncate while preserving structure
                truncated_context = full_context[:context_length_limit]
                # Try to cut at a section boundary
                last_section = truncated_context.rfind("===")
                if last_section > context_length_limit * 0.7:
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
        """Save interaction to both retriever memory and conversation history"""
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

    def save(self, context: str, output: str):
        """Legacy save method for backward compatibility"""
        self.save_interaction(context, output)

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

    @property
    def retriever(self):
        """Legacy property for backward compatibility"""
        return self.retriever_memory


# Original MemoryManager class for full backward compatibility
class MemoryManager:
    """Original MemoryManager class with enhanced functionality"""

    def __init__(self, external_urls: Optional[List[str]] = None):
        # Use the enhanced manager internally
        self._enhanced = EnhancedMemoryManager(external_urls)

        # Expose all necessary attributes
        self.vector_store = self._enhanced.vector_store
        self.retriever_memory = self._enhanced.retriever_memory
        self.composite_retriever = self._enhanced.composite_retriever
        self.conversation_history = self._enhanced.conversation_history  # Expose conversation history
        self.stats = self._enhanced.stats  # Expose stats
        self.external_urls = self._enhanced.external_urls

        # Try to get external vector store reference for backward compatibility
        self.external_vector_store = None
        if external_urls:
            try:
                self.external_vector_store = get_external_vector_store(external_urls)
            except:
                pass

    def get_augmented_context(self,
                              query: str,
                              k: int = 4,
                              include_conversation_history: bool = True,
                              context_length_limit: int = 4000) -> str:
        """Get augmented context using enhanced manager"""
        return self._enhanced.get_augmented_context(
            query,
            include_conversation_history=include_conversation_history,
            context_length_limit=context_length_limit,
            k=k
        )

    def save(self, context: str, output: str):
        """Save interaction to memory"""
        self._enhanced.save(context, output)

    def save_interaction(self, user_input: str, assistant_output: str, metadata: Dict[str, Any] = None):
        """Save interaction with metadata"""
        self._enhanced.save_interaction(user_input, assistant_output, metadata)

    @property
    def retriever(self):
        """Property for backward compatibility"""
        return self.retriever_memory

    def get_stats(self):
        """Get basic statistics"""
        return self._enhanced.get_memory_stats()

    def get_memory_stats(self):
        """Get memory statistics"""
        return self._enhanced.get_memory_stats()


# Factory functions
def create_memory_manager(enhanced: bool = True, **kwargs):
    """Create a memory manager"""
    if enhanced:
        return EnhancedMemoryManager(**kwargs)
    else:
        return MemoryManager(**kwargs)


def create_enhanced_memory_manager(**kwargs):
    """Create enhanced memory manager"""
    return EnhancedMemoryManager(**kwargs)


# Usage example
if __name__ == "__main__":
    # Test with enhanced interface
    print("Testing Enhanced Memory Manager...")

    try:
        memory_manager = MemoryManager()  # Test the legacy interface

        # Test retrieval
        query = "What is machine learning?"
        context = memory_manager.get_augmented_context(query, context_length_limit=2000)

        print(f"Context length: {len(context)}")
        print("Context preview:")
        print(context[:300] + "..." if len(context) > 300 else context)

        # Test saving
        response = "Machine learning is a subset of AI..."
        memory_manager.save(query, response)

        # Show stats
        print("\nMemory Stats:")
        stats = memory_manager.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")

        print("\nMemory Manager test completed successfully!")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"Test failed: {e}")
