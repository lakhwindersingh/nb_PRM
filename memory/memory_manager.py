# memory/memory_manager.py
import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from enum import Enum

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


class ResponseStyle(Enum):
    """Response enhancement styles"""
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
    def enhance_prompt(prompt: str, style: ResponseStyle = ResponseStyle.DETAILED) -> str:
        """Enhance a prompt based on the specified style"""

        if style == ResponseStyle.AUTO:
            style = PromptEnhancer._detect_style(prompt)

        enhancers = {
            ResponseStyle.DETAILED: PromptEnhancer._enhance_detailed,
            ResponseStyle.ANALYTICAL: PromptEnhancer._enhance_analytical,
            ResponseStyle.HOWTO: PromptEnhancer._enhance_howto,
            ResponseStyle.COMPARISON: PromptEnhancer._enhance_comparison,
            ResponseStyle.CREATIVE: PromptEnhancer._enhance_creative,
            ResponseStyle.RESEARCH: PromptEnhancer._enhance_research,
            ResponseStyle.CODING: PromptEnhancer._enhance_coding,
            ResponseStyle.EXPLANATION: PromptEnhancer._enhance_explanation
        }

        enhancer = enhancers.get(style, enhancers[ResponseStyle.DETAILED])
        return enhancer(prompt)

    @staticmethod
    def _detect_style(prompt: str) -> ResponseStyle:
        """Auto-detect the best enhancement style for a prompt"""
        prompt_lower = prompt.lower()

        if any(word in prompt_lower for word in ['how to', 'steps', 'process', 'guide', 'tutorial']):
            return ResponseStyle.HOWTO
        elif any(word in prompt_lower for word in ['compare', 'vs', 'versus', 'difference', 'better']):
            return ResponseStyle.COMPARISON
        elif any(word in prompt_lower for word in ['analyze', 'analysis', 'evaluate', 'assess', 'examine']):
            return ResponseStyle.ANALYTICAL
        elif any(word in prompt_lower for word in ['create', 'design', 'write', 'compose', 'generate']):
            return ResponseStyle.CREATIVE
        elif any(word in prompt_lower for word in ['research', 'study', 'investigate', 'find out', 'sources']):
            return ResponseStyle.RESEARCH
        elif any(word in prompt_lower for word in ['code', 'program', 'implement', 'algorithm', 'function']):
            return ResponseStyle.CODING
        elif any(word in prompt_lower for word in ['explain', 'what is', 'definition', 'meaning', 'understand']):
            return ResponseStyle.EXPLANATION
        else:
            return ResponseStyle.DETAILED

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

**Quality Standards:**
- Provide comprehensive coverage without unnecessary repetition
- Use clear headings and logical structure
- Include specific examples and concrete details
- Maintain accuracy and cite sources when relevant
- Address multiple aspects and perspectives
- End with actionable insights or next steps

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
6. **Evaluation** - Strengths, weaknesses, opportunities, threats
7. **Conclusions & Recommendations** - Evidence-based outcomes

**Analysis Standards:**
- Use logical reasoning and evidence-based conclusions
- Consider multiple perspectives and potential biases
- Identify patterns, trends, and relationships
- Distinguish between facts and interpretations
- Provide balanced evaluation of pros and cons
- Support all claims with specific evidence

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
3. **Preparation** - Setup steps and initial requirements
4. **Step-by-Step Instructions** - Detailed, numbered process
5. **Tips & Best Practices** - Professional insights and optimizations
6. **Common Pitfalls** - What to avoid and troubleshooting
7. **Verification** - How to confirm success and quality checks

**Instruction Quality:**
- Make each step clear, specific, and actionable
- Use simple language and avoid jargon where possible
- Include time estimates and difficulty levels
- Provide alternatives for different situations
- Add safety warnings or important notes
- Include examples or visual descriptions when helpful

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
3. **Option Profiles** - Detailed description of each alternative
4. **Feature Comparison** - Side-by-side analysis of key attributes
5. **Strengths & Weaknesses** - Pros and cons for each option
6. **Use Case Analysis** - Best scenarios for each choice
7. **Final Verdict** - Recommendations with clear reasoning

**Comparison Standards:**
- Use consistent criteria across all options
- Provide objective, fact-based analysis
- Include both quantitative and qualitative factors
- Consider different user needs and contexts
- Present information in easy-to-compare formats
- Give clear rationale for recommendations

Deliver a balanced comparison that helps readers make informed decisions based on their specific needs.
"""

    @staticmethod
    def _enhance_creative(prompt: str) -> str:
        return f"""
You are a creative expert and innovative thinker. Approach this with imagination and originality:

**Creative Challenge:** {prompt}

**Creative Structure:**
1. **Creative Vision** - Innovative concept and inspiration
2. **Conceptual Framework** - Underlying ideas and themes
3. **Creative Development** - Detailed exploration of possibilities
4. **Multiple Approaches** - Different creative directions and styles
5. **Concrete Examples** - Specific illustrations and demonstrations
6. **Implementation Ideas** - How to bring concepts to life
7. **Variations & Extensions** - Additional creative possibilities

**Creative Standards:**
- Think outside conventional boundaries
- Generate original and innovative ideas
- Provide rich, detailed creative content
- Include multiple creative alternatives
- Balance creativity with practicality
- Use vivid descriptions and compelling examples

Be imaginative, original, and inspiring while providing practical guidance for implementation.
"""

    @staticmethod
    def _enhance_research(prompt: str) -> str:
        return f"""
You are a research expert providing comprehensive investigation results. Structure as research output:

**Research Question:** {prompt}

**Research Report Structure:**
1. **Research Objective** - What question is being investigated
2. **Background Context** - Relevant foundational information
3. **Methodology** - Research approach and information sources
4. **Key Findings** - Main discoveries and data points
5. **Supporting Evidence** - Sources, studies, and documentation
6. **Analysis & Interpretation** - What the findings mean
7. **Research Implications** - Significance and applications

**Research Standards:**
- Base conclusions on credible sources and evidence
- Distinguish between established facts and emerging theories
- Acknowledge limitations and areas of uncertainty
- Include recent developments and current state of knowledge
- Reference authoritative sources and studies
- Maintain objectivity and avoid unsupported claims

Provide research-quality information that would be valuable for academic or professional purposes.
"""

    @staticmethod
    def _enhance_coding(prompt: str) -> str:
        return f"""
You are an expert software engineer and code architect. Provide comprehensive coding guidance:

**Programming Request:** {prompt}

**Development Structure:**
1. **Problem Analysis** - Break down requirements and constraints
2. **Solution Design** - Architecture, approach, and technology choices
3. **Implementation** - Complete, working code with clear comments
4. **Code Walkthrough** - Explanation of key logic and components
5. **Testing & Validation** - Test cases and quality assurance
6. **Optimization** - Performance improvements and best practices
7. **Documentation** - Usage examples and maintenance notes

**Coding Standards:**
- Write clean, readable, maintainable code
- Follow language-specific best practices and conventions
- Include comprehensive error handling
- Add meaningful comments and documentation
- Provide working examples and test cases
- Consider scalability and performance implications
- Suggest improvements and alternative approaches

Deliver production-quality code with thorough explanation and professional development practices.
"""

    @staticmethod
    def _enhance_explanation(prompt: str) -> str:
        return f"""
You are an expert educator providing clear, comprehensive explanations. Structure your teaching:

**Topic to Explain:** {prompt}

**Educational Structure:**
1. **Introduction** - Hook and overview of what will be learned
2. **Key Concepts** - Essential definitions and foundational ideas
3. **Detailed Explanation** - Step-by-step breakdown of the topic
4. **Examples & Illustrations** - Concrete demonstrations and analogies
5. **Common Misconceptions** - What people often get wrong and why
6. **Practical Applications** - Real-world uses and relevance
7. **Summary & Next Steps** - Key takeaways and further learning

**Teaching Standards:**
- Start with basics and build complexity gradually
- Use clear, jargon-free language with definitions
- Include multiple examples and analogies
- Address different learning styles and perspectives
- Anticipate and answer common questions
- Connect to prior knowledge and real-world experiences

Create an explanation that would help someone truly understand the topic, not just memorize facts.
"""


class EnhancedMemoryManager:
    """
    Enhanced memory manager with built-in prompt enhancement capabilities
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

    def get_enhanced_context(self,
                             query: str,
                             enhancement_style: ResponseStyle = ResponseStyle.AUTO,
                             include_conversation_history: bool = True,
                             context_length_limit: int = 4000,
                             k: int = None,
                             return_components: bool = False) -> str:
        """
        Get enhanced prompt with augmented context from multiple knowledge sources

        Args:
            query: The user's query
            enhancement_style: Style of prompt enhancement
            include_conversation_history: Whether to include conversation history
            context_length_limit: Maximum context length
            k: Number of documents to retrieve
            return_components: If True, returns tuple of (enhanced_prompt, context, full_prompt)

        Returns:
            Enhanced prompt with context, or tuple if return_components=True
        """
        try:
            self.stats['total_retrievals'] += 1
            self.stats['last_retrieval'] = datetime.now().isoformat()

            # Enhance the prompt
            enhanced_prompt = PromptEnhancer.enhance_prompt(query, enhancement_style)

            # Track enhancement stats
            self.stats['enhancement_stats']['total_enhancements'] += 1
            style_name = enhancement_style.value if enhancement_style != ResponseStyle.AUTO else 'auto_detected'
            self.stats['enhancement_stats']['by_style'][style_name] = \
                self.stats['enhancement_stats']['by_style'].get(style_name, 0) + 1

            # Get augmented context
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
            retrieved_context = "\n\n".join(context_parts)

            # Create full prompt
            full_prompt = self._create_full_prompt(enhanced_prompt, retrieved_context, context_length_limit)

            logger.info(f"Enhanced prompt and retrieved context for query: '{query[:50]}...'")

            if return_components:
                return enhanced_prompt, retrieved_context, full_prompt
            else:
                return full_prompt

        except Exception as e:
            logger.error(f"Failed to get enhanced context: {e}")
            # Fallback
            if return_components:
                return query, "", query
            else:
                return query

    def _create_full_prompt(self, enhanced_prompt: str, context: str, length_limit: int) -> str:
        """Create the complete prompt combining enhanced prompt with context"""

        prompt_parts = []

        # Add context if available
        if context and context.strip():
            prompt_parts.append(f"CONTEXT INFORMATION:\n{context}")

        # Add enhanced prompt
        prompt_parts.append(f"REQUEST:\n{enhanced_prompt}")

        # Add final instructions
        prompt_parts.append("""
RESPONSE GUIDELINES:
- Use the context information to inform your response when relevant
- Follow the structured format specified in the request
- Provide comprehensive, well-organized answers
- Include examples and concrete details where helpful
- If context is insufficient for complete answers, acknowledge this
- Maintain accuracy and cite context sources when making specific claims
""")

        # Combine and check length
        full_prompt = "\n\n".join(prompt_parts)

        # Truncate if necessary while preserving structure
        if len(full_prompt) > length_limit:
            # Try to preserve the enhanced prompt and truncate context
            context_limit = length_limit - len(enhanced_prompt) - 500  # Leave room for instructions
            if context_limit > 0 and context:
                truncated_context = context[:context_limit] + "\n\n[... context truncated ...]"
                full_prompt = self._create_full_prompt(enhanced_prompt, truncated_context, length_limit)
            else:
                full_prompt = full_prompt[:length_limit] + "\n\n[... prompt truncated ...]"

        return full_prompt

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

    # Convenience methods for different enhancement styles
    def get_detailed_context(self, query: str, **kwargs):
        """Get detailed, comprehensive response context"""
        return self.get_enhanced_context(query, ResponseStyle.DETAILED, **kwargs)

    def get_analytical_context(self, query: str, **kwargs):
        """Get analytical response context"""
        return self.get_enhanced_context(query, ResponseStyle.ANALYTICAL, **kwargs)

    def get_howto_context(self, query: str, **kwargs):
        """Get how-to/instructional response context"""
        return self.get_enhanced_context(query, ResponseStyle.HOWTO, **kwargs)

    def get_comparison_context(self, query: str, **kwargs):
        """Get comparison response context"""
        return self.get_enhanced_context(query, ResponseStyle.COMPARISON, **kwargs)

    def get_coding_context(self, query: str, **kwargs):
        """Get coding/programming response context"""
        return self.get_enhanced_context(query, ResponseStyle.CODING, **kwargs)

    def get_creative_context(self, query: str, **kwargs):
        """Get creative response context"""
        return self.get_enhanced_context(query, ResponseStyle.CREATIVE, **kwargs)

    def get_research_context(self, query: str, **kwargs):
        """Get research response context"""
        return self.get_enhanced_context(query, ResponseStyle.RESEARCH, **kwargs)

    def get_explanation_context(self, query: str, **kwargs):
        """Get explanation response context"""
        return self.get_enhanced_context(query, ResponseStyle.EXPLANATION, **kwargs)

    # Legacy compatibility methods
    def get_augmented_context(self, query: str, **kwargs) -> str:
        """Legacy method - returns basic augmented context without enhancement"""
        try:
            context_parts = []

            # Get knowledge base context
            if self.composite_retriever:
                try:
                    knowledge_context = self.composite_retriever.get_combined_context(
                        query, k=kwargs.get('k', self.retrieval_k)
                    )
                    if knowledge_context:
                        context_parts.append(knowledge_context)
                except Exception as e:
                    logger.warning(f"Composite retriever failed: {e}")
                    fallback_context = self._get_fallback_context(query, k=kwargs.get('k', 3))
                    if fallback_context:
                        context_parts.append(fallback_context)

            return "\n\n".join(context_parts)

        except Exception as e:
            logger.error(f"Failed to get augmented context: {e}")
            return "Context retrieval failed."

    def save_interaction(self,
                         user_input: str,
                         assistant_output: str,
                         metadata: Optional[Dict[str, Any]] = None):
        """
        Save interaction to both retriever memory and conversation history
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

    def save(self, context: str, output: str):
        """Legacy save method for backward compatibility"""
        self.save_interaction(context, output)

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

    @property
    def retriever(self):
        """Legacy property for backward compatibility"""
        return self.retriever_memory


# Original MemoryManager class for full backward compatibility
class MemoryManager:
    """Original MemoryManager class with enhanced functionality"""

    def __init__(self, external_urls: Optional[List[str]] = None):
        # Initialize enhanced manager internally
        self.enhanced_manager = EnhancedMemoryManager(external_urls)

        # Expose key attributes for compatibility
        self.vector_store = self.enhanced_manager.vector_store
        self.retriever_memory = self.enhanced_manager.retriever_memory
        self.external_vector_store = None
        self.composite_retriever = self.enhanced_manager.composite_retriever

        # Try to get external vector store reference
        if external_urls:
            try:
                self.external_vector_store = get_external_vector_store(external_urls)
            except:
                pass

    def get_augmented_context(self, query, k=4):
        """Get augmented context using composite retriever"""
        return self.enhanced_manager.get_augmented_context(query, k=k)

    def save(self, context, output):
        """Save interaction to memory"""
        self.enhanced_manager.save(context, output)

    @property
    def retriever(self):
        """Property for backward compatibility"""
        return self.retriever_memory

    def get_stats(self):
        """Get basic statistics"""
        return self.enhanced_manager.get_memory_stats()

    # New enhanced methods available in legacy class
    def get_enhanced_context(self, query: str, style: str = "auto", **kwargs):
        """Get enhanced context with prompt enhancement"""
        style_enum = ResponseStyle(style) if isinstance(style, str) else style
        return self.enhanced_manager.get_enhanced_context(query, style_enum, **kwargs)

    def get_detailed_context(self, query: str, **kwargs):
        """Get detailed response context"""
        return self.enhanced_manager.get_detailed_context(query, **kwargs)

    def get_coding_context(self, query: str, **kwargs):
        """Get coding response context"""
        return self.enhanced_manager.get_coding_context(query, **kwargs)


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
        memory_manager = EnhancedMemoryManager()

        # Test different enhancement styles
        queries = [
            ("What is machine learning?", ResponseStyle.EXPLANATION),
            ("How to implement a binary search?", ResponseStyle.CODING),
            ("Compare Python and Java", ResponseStyle.COMPARISON),
            ("Analyze the impact of AI on society", ResponseStyle.ANALYTICAL)
        ]

        for query, style in queries:
            print(f"\n=== {style.value.upper()} ENHANCEMENT ===")
            print(f"Query: {query}")

            enhanced_prompt, context, full_prompt = memory_manager.get_enhanced_context(
                query, style, return_components=True
            )

            print(f"Enhanced Prompt Length: {len(enhanced_prompt)}")
            print(f"Context Length: {len(context)}")
            print(f"Full Prompt Length: {len(full_prompt)}")

            # Show preview
            print("Enhanced Prompt Preview:")
            print(enhanced_prompt[:300] + "..." if len(enhanced_prompt) > 300 else enhanced_prompt)
            print("-" * 80)

        # Test auto-detection
        print("\n=== AUTO ENHANCEMENT TEST ===")
        auto_query = "How do I create a REST API in Python?"
        full_prompt = memory_manager.get_enhanced_context(auto_query, ResponseStyle.AUTO)
        print(f"Auto-detected enhancement for: {auto_query}")
        print(f"Full prompt length: {len(full_prompt)}")

        # Show stats
        print("\n=== MEMORY STATISTICS ===")
        stats = memory_manager.get_memory_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")

        print("\nEnhanced Memory Manager test completed successfully!")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"Test failed: {e}")
