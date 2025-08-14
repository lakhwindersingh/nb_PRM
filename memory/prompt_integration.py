# memory/prompt_integration.py
from memory.memory_manager import MemoryManager
from utils.quick_enhance import quick_enhance, smart_enhance, enhance_with_context
import logging

logger = logging.getLogger(__name__)


class PromptEnhancedMemoryManager:
    """Simple wrapper that adds prompt enhancement to existing MemoryManager"""

    def __init__(self, external_urls=None):
        self.memory_manager = MemoryManager(external_urls)
        self.enhancement_stats = {
            'total_enhancements': 0,
            'by_style': {}
        }

    def get_enhanced_response_context(self,
                                      query: str,
                                      enhancement_style: str = "auto",
                                      include_history: bool = True,
                                      context_limit: int = 4000,
                                      **kwargs) -> tuple:
        """
        Get enhanced prompt and retrieve relevant context

        Returns:
            Tuple of (enhanced_prompt, retrieved_context, full_prompt)
        """
        try:
            # Enhance the prompt
            if enhancement_style == "auto":
                enhanced_prompt = smart_enhance(query)
            else:
                enhanced_prompt = quick_enhance(query, enhancement_style)

            # Track enhancement stats
            self.enhancement_stats['total_enhancements'] += 1
            style_key = enhancement_style if enhancement_style != "auto" else "smart_auto"
            self.enhancement_stats['by_style'][style_key] = self.enhancement_stats['by_style'].get(style_key, 0) + 1

            # Get retrieved context using original query (for better matching)
            retrieved_context = self.memory_manager.get_augmented_context(query, k=kwargs.get('k', 4))

            # Create full prompt combining enhanced prompt with retrieved context
            full_prompt = self._create_complete_prompt(enhanced_prompt, retrieved_context, **kwargs)

            logger.info(f"Enhanced prompt and retrieved context for: '{query[:50]}...'")

            return enhanced_prompt, retrieved_context, full_prompt

        except Exception as e:
            logger.error(f"Failed to enhance prompt and get context: {e}")
            # Fallback to basic functionality
            context = self.memory_manager.get_augmented_context(query)
            return query, context, f"{query}\n\nContext:\n{context}"

    def _create_complete_prompt(self, enhanced_prompt: str, context: str, **kwargs) -> str:
        """Create a complete prompt with context and instructions"""

        system_instruction = kwargs.get('system_instruction',
                                        "You are a knowledgeable AI assistant. Use the provided context to inform your response, "
                                        "but don't limit yourself to only the context information. Follow the structured format "
                                        "requested in the user prompt."
                                        )

        prompt_parts = []

        # System instruction
        prompt_parts.append(f"SYSTEM: {system_instruction}")

        # Retrieved context (if available)
        if context and context.strip():
            prompt_parts.append(f"CONTEXT:\n{context}")

        # Enhanced user prompt
        prompt_parts.append(f"REQUEST:\n{enhanced_prompt}")

        # Additional instructions
        additional_instructions = [
            "- Provide a well-structured, comprehensive response",
            "- Use the context information where relevant",
            "- If context is insufficient, acknowledge this and provide the best response possible",
            "- Follow any specific formatting requirements mentioned in the request"
        ]

        if kwargs.get('require_sources', False):
            additional_instructions.append("- Cite sources from the context when making specific claims")

        if kwargs.get('encourage_examples', True):
            additional_instructions.append("- Include concrete examples where helpful")

        prompt_parts.append("INSTRUCTIONS:\n" + "\n".join(additional_instructions))

        return "\n\n".join(prompt_parts)

    def save_interaction(self, query: str, response: str, metadata: dict = None):
        """Save interaction using underlying memory manager"""
        self.memory_manager.save(query, response)

        if metadata:
            # Could extend to save metadata if needed
            logger.info(f"Saved interaction with metadata: {list(metadata.keys())}")

    def get_stats(self) -> dict:
        """Get combined stats from memory manager and enhancement"""
        base_stats = self.memory_manager.get_stats()
        return {
            **base_stats,
            'enhancement_stats': self.enhancement_stats
        }

    # Convenience methods for common enhancement patterns
    def get_detailed_response_context(self, query: str, **kwargs):
        """Get detailed, comprehensive response context"""
        return self.get_enhanced_response_context(query, "detailed", **kwargs)

    def get_analytical_response_context(self, query: str, **kwargs):
        """Get analytical response context"""
        return self.get_enhanced_response_context(query, "analytical", **kwargs)

    def get_howto_response_context(self, query: str, **kwargs):
        """Get how-to/instructional response context"""
        return self.get_enhanced_response_context(query, "howto", **kwargs)

    def get_comparison_response_context(self, query: str, **kwargs):
        """Get comparison response context"""
        return self.get_enhanced_response_context(query, "comparison", **kwargs)

    def get_coding_response_context(self, query: str, **kwargs):
        """Get coding/programming response context"""
        return self.get_enhanced_response_context(query, "coding", **kwargs)


# Convenience factory function
def create_enhanced_memory_manager(external_urls=None):
    """Create a prompt-enhanced memory manager"""
    return PromptEnhancedMemoryManager(external_urls)
