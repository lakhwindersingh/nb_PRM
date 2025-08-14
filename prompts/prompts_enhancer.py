# prompts/prompt_enhancer.py
import re
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class ResponseFormat(Enum):
    """Different structured response formats"""
    DETAILED = "detailed"
    ANALYTICAL = "analytical"
    STEP_BY_STEP = "step_by_step"
    COMPARATIVE = "comparative"
    PROBLEM_SOLVING = "problem_solving"
    CREATIVE = "creative"
    RESEARCH = "research"
    TECHNICAL = "technical"


class DetailLevel(Enum):
    """Detail levels for responses"""
    BRIEF = "brief"
    MODERATE = "moderate"
    COMPREHENSIVE = "comprehensive"
    EXHAUSTIVE = "exhaustive"


@dataclass
class PromptContext:
    """Context information for prompt enhancement"""
    domain: Optional[str] = None
    audience: Optional[str] = None
    purpose: Optional[str] = None
    constraints: Optional[List[str]] = None
    examples_needed: bool = False
    citations_needed: bool = False


class PromptEnhancer:
    """Advanced prompt enhancement system for structured responses"""

    def __init__(self):
        self.format_templates = self._load_format_templates()
        self.structure_patterns = self._load_structure_patterns()
        self.enhancement_rules = self._load_enhancement_rules()

    def enhance_prompt(self,
                       original_prompt: str,
                       response_format: ResponseFormat = ResponseFormat.DETAILED,
                       detail_level: DetailLevel = DetailLevel.COMPREHENSIVE,
                       context: Optional[PromptContext] = None,
                       custom_instructions: Optional[List[str]] = None) -> str:
        """
        Enhance a prompt to get more structured and detailed responses

        Args:
            original_prompt: The original user prompt
            response_format: Desired response structure format
            detail_level: Level of detail required
            context: Additional context information
            custom_instructions: Custom formatting instructions

        Returns:
            Enhanced prompt string
        """
        try:
            # Analyze the original prompt
            prompt_analysis = self._analyze_prompt(original_prompt)

            # Build enhanced prompt components
            components = []

            # Add role and context setting
            components.append(self._build_role_context(context, prompt_analysis))

            # Add structured response instructions
            components.append(self._build_response_format(response_format, detail_level))

            # Add the enhanced original prompt
            components.append(self._build_enhanced_query(original_prompt, prompt_analysis))

            # Add specific formatting instructions
            components.append(self._build_formatting_instructions(
                response_format, detail_level, context, custom_instructions
            ))

            # Add quality assurance instructions
            components.append(self._build_quality_instructions(detail_level, context))

            # Combine all components
            enhanced_prompt = "\n\n".join(filter(None, components))

            logger.info(f"Enhanced prompt from {len(original_prompt)} to {len(enhanced_prompt)} characters")
            return enhanced_prompt

        except Exception as e:
            logger.error(f"Failed to enhance prompt: {e}")
            return original_prompt  # Fallback to original

    def _analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        """Analyze the original prompt to understand its nature"""
        analysis = {
            'type': 'general',
            'complexity': 'medium',
            'requires_examples': False,
            'requires_citations': False,
            'is_question': '?' in prompt,
            'is_creative': any(word in prompt.lower() for word in ['create', 'write', 'compose', 'design', 'imagine']),
            'is_analytical': any(
                word in prompt.lower() for word in ['analyze', 'compare', 'evaluate', 'assess', 'examine']),
            'is_technical': any(
                word in prompt.lower() for word in ['implement', 'code', 'algorithm', 'technical', 'system']),
            'is_research': any(
                word in prompt.lower() for word in ['research', 'study', 'investigate', 'explore', 'find']),
            'word_count': len(prompt.split()),
            'topics': self._extract_topics(prompt)
        }

        # Determine complexity
        if analysis['word_count'] > 50 or len(analysis['topics']) > 3:
            analysis['complexity'] = 'high'
        elif analysis['word_count'] < 10:
            analysis['complexity'] = 'low'

        # Determine if examples/citations needed
        analysis['requires_examples'] = any(
            word in prompt.lower() for word in ['example', 'instance', 'demonstrate', 'show'])
        analysis['requires_citations'] = any(
            word in prompt.lower() for word in ['source', 'research', 'study', 'evidence', 'cite'])

        return analysis

    def _extract_topics(self, prompt: str) -> List[str]:
        """Extract key topics from the prompt"""
        # Simple keyword extraction - could be enhanced with NLP
        words = re.findall(r'\b[A-Za-z]{3,}\b', prompt.lower())
        # Filter common words
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our',
                      'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way',
                      'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'}
        topics = [word for word in words if word not in stop_words and len(word) > 3]
        return list(set(topics))[:5]  # Return up to 5 unique topics

    def _build_role_context(self, context: Optional[PromptContext], analysis: Dict[str, Any]) -> str:
        """Build role and context setting for the AI"""
        role_parts = []

        # Determine appropriate role
        if analysis.get('is_technical'):
            role_parts.append("You are an expert technical consultant with deep knowledge across multiple domains.")
        elif analysis.get('is_research'):
            role_parts.append("You are a knowledgeable researcher with access to comprehensive information.")
        elif analysis.get('is_creative'):
            role_parts.append("You are a creative expert with strong analytical and imaginative capabilities.")
        elif analysis.get('is_analytical'):
            role_parts.append("You are an analytical expert capable of deep critical thinking and evaluation.")
        else:
            role_parts.append("You are a knowledgeable assistant with expertise across multiple domains.")

        # Add context-specific role enhancements
        if context:
            if context.domain:
                role_parts.append(f"You specialize in {context.domain}.")
            if context.audience:
                role_parts.append(f"You are communicating with {context.audience}.")
            if context.purpose:
                role_parts.append(f"Your goal is to {context.purpose}.")

        role_parts.append("Provide thorough, accurate, and well-structured responses.")

        return " ".join(role_parts)

    def _build_response_format(self, format_type: ResponseFormat, detail_level: DetailLevel) -> str:
        """Build response format instructions"""
        format_instructions = {
            ResponseFormat.DETAILED: """
Structure your response with clear sections and comprehensive coverage:
1. **Executive Summary** - Key points and main conclusions
2. **Detailed Analysis** - In-depth exploration of the topic
3. **Supporting Evidence** - Facts, data, and examples
4. **Implications** - Broader context and significance
5. **Conclusion** - Summary and final thoughts
""",

            ResponseFormat.ANALYTICAL: """
Provide a structured analytical response:
1. **Problem Definition** - Clearly state what is being analyzed
2. **Methodology** - Approach and framework used
3. **Analysis** - Systematic examination with evidence
4. **Findings** - Key discoveries and insights
5. **Recommendations** - Actionable conclusions
""",

            ResponseFormat.STEP_BY_STEP: """
Break down your response into clear, actionable steps:
1. **Overview** - What will be accomplished
2. **Prerequisites** - What's needed before starting
3. **Step-by-Step Process** - Detailed instructions with sub-steps
4. **Verification** - How to check if each step is successful
5. **Troubleshooting** - Common issues and solutions
""",

            ResponseFormat.COMPARATIVE: """
Structure your response as a comprehensive comparison:
1. **Introduction** - What is being compared and why
2. **Comparison Framework** - Criteria and methodology
3. **Side-by-Side Analysis** - Detailed point-by-point comparison
4. **Strengths and Weaknesses** - Pros and cons of each option
5. **Conclusion** - Best choice or recommendation with reasoning
""",

            ResponseFormat.PROBLEM_SOLVING: """
Approach this as a problem-solving exercise:
1. **Problem Statement** - Clear definition of the challenge
2. **Root Cause Analysis** - Understanding underlying issues
3. **Solution Options** - Multiple approaches with trade-offs
4. **Recommended Solution** - Best approach with detailed implementation
5. **Risk Mitigation** - Potential issues and preventive measures
""",

            ResponseFormat.CREATIVE: """
Provide a creative and comprehensive response:
1. **Creative Vision** - Overall concept and inspiration
2. **Detailed Development** - Thorough exploration of ideas
3. **Examples and Illustrations** - Concrete demonstrations
4. **Variations and Alternatives** - Different approaches or styles
5. **Implementation Guidance** - How to bring ideas to life
""",

            ResponseFormat.RESEARCH: """
Present your response as a research summary:
1. **Research Objective** - What question is being investigated
2. **Background Context** - Relevant foundational information
3. **Key Findings** - Main discoveries and data points
4. **Supporting Evidence** - Sources, studies, and examples
5. **Research Implications** - Significance and applications
""",

            ResponseFormat.TECHNICAL: """
Provide a technical deep-dive:
1. **Technical Overview** - High-level architecture or concept
2. **Detailed Specifications** - Technical requirements and details
3. **Implementation Details** - Code, configurations, or procedures
4. **Best Practices** - Industry standards and recommendations
5. **Testing and Validation** - Quality assurance approaches
"""
        }

        base_format = format_instructions.get(format_type, format_instructions[ResponseFormat.DETAILED])

        # Add detail level instructions
        detail_instructions = {
            DetailLevel.BRIEF: "Keep each section concise but informative (2-3 sentences per section).",
            DetailLevel.MODERATE: "Provide solid detail in each section (1-2 paragraphs per section).",
            DetailLevel.COMPREHENSIVE: "Give thorough coverage of each section (2-4 paragraphs per section).",
            DetailLevel.EXHAUSTIVE: "Provide exhaustive detail in each section (multiple paragraphs with sub-points)."
        }

        return base_format + "\n" + detail_instructions[detail_level]

    def _build_enhanced_query(self, original_prompt: str, analysis: Dict[str, Any]) -> str:
        """Build the enhanced version of the original query"""
        enhanced_parts = [f"**Main Query:** {original_prompt}"]

        # Add clarifying questions based on analysis
        if analysis['complexity'] == 'high':
            enhanced_parts.append("Please address all aspects of this multi-faceted question.")

        if analysis['topics']:
            enhanced_parts.append(f"Key areas to cover include: {', '.join(analysis['topics'])}.")

        if analysis.get('requires_examples'):
            enhanced_parts.append("Include specific examples and concrete illustrations.")

        if analysis.get('requires_citations'):
            enhanced_parts.append("Reference credible sources and provide citations where applicable.")

        return " ".join(enhanced_parts)

    def _build_formatting_instructions(self,
                                       format_type: ResponseFormat,
                                       detail_level: DetailLevel,
                                       context: Optional[PromptContext],
                                       custom_instructions: Optional[List[str]]) -> str:
        """Build specific formatting and presentation instructions"""
        instructions = []

        # Base formatting rules
        instructions.append("**Formatting Requirements:**")
        instructions.append("- Use clear headings and subheadings with markdown formatting")
        instructions.append("- Include bullet points and numbered lists where appropriate")
        instructions.append("- Use **bold** for emphasis and *italics* for definitions")
        instructions.append("- Separate major sections with clear breaks")

        # Context-specific instructions
        if context:
            if context.examples_needed:
                instructions.append("- Provide concrete examples for each major point")
            if context.citations_needed:
                instructions.append("- Include citations in [Source: Description] format")
            if context.constraints:
                for constraint in context.constraints:
                    instructions.append(f"- Constraint: {constraint}")

        # Custom instructions
        if custom_instructions:
            instructions.append("**Additional Requirements:**")
            for instruction in custom_instructions:
                instructions.append(f"- {instruction}")

        return "\n".join(instructions)

    def _build_quality_instructions(self, detail_level: DetailLevel, context: Optional[PromptContext]) -> str:
        """Build quality assurance and completeness instructions"""
        quality_parts = []

        quality_parts.append("**Quality Standards:**")
        quality_parts.append("- Ensure accuracy and factual correctness")
        quality_parts.append("- Maintain logical flow and coherent structure")
        quality_parts.append("- Use clear, professional language appropriate for the audience")

        if detail_level in [DetailLevel.COMPREHENSIVE, DetailLevel.EXHAUSTIVE]:
            quality_parts.append("- Provide comprehensive coverage without unnecessary repetition")
            quality_parts.append("- Include relevant background context and implications")

        quality_parts.append("- End with a brief summary if the response is lengthy")
        quality_parts.append("- Double-check that all parts of the original question are addressed")

        return "\n".join(quality_parts)

    def _load_format_templates(self) -> Dict[str, str]:
        """Load format templates (could be from files in production)"""
        return {}  # Placeholder - could load from external files

    def _load_structure_patterns(self) -> Dict[str, List[str]]:
        """Load structure patterns (could be from files in production)"""
        return {}  # Placeholder - could load from external files

    def _load_enhancement_rules(self) -> List[Dict[str, Any]]:
        """Load enhancement rules (could be from files in production)"""
        return []  # Placeholder - could load from external files


class PromptTemplate:
    """Template-based prompt enhancement for common scenarios"""

    def __init__(self):
        self.templates = {
            'explanation': """
You are an expert educator and communicator. Provide a comprehensive explanation that:

1. **Starts with a clear definition** - Define key terms and concepts
2. **Provides context** - Background information and why this matters
3. **Explains the main content** - Detailed explanation broken into logical sections
4. **Includes examples** - Concrete illustrations and real-world applications
5. **Addresses implications** - What this means and why it's significant
6. **Concludes with summary** - Key takeaways and next steps

**Topic:** {query}

Structure your response with clear headings, use examples throughout, and ensure each section flows logically to the next. Aim for comprehensive coverage while maintaining clarity.
""",

            'how_to': """
You are an expert instructor providing step-by-step guidance. Create a comprehensive how-to guide that:

1. **Overview** - What will be accomplished and why
2. **Prerequisites** - Required knowledge, tools, or preparation
3. **Materials/Requirements** - Everything needed before starting
4. **Step-by-Step Instructions** - Detailed process with clear numbering
5. **Tips and Best Practices** - Professional insights and optimization
6. **Troubleshooting** - Common issues and solutions
7. **Verification** - How to confirm success at each stage

**Request:** {query}

Make each step actionable and specific. Include warnings for potential pitfalls and provide alternative approaches where relevant.
""",

            'analysis': """
You are an expert analyst providing comprehensive evaluation. Structure your analysis as follows:

1. **Executive Summary** - Key findings and conclusions upfront
2. **Scope and Methodology** - What you're analyzing and how
3. **Detailed Analysis** - Systematic examination of each component
4. **Evidence and Data** - Supporting information and sources
5. **Strengths and Weaknesses** - Balanced evaluation of pros/cons
6. **Implications** - Broader significance and consequences
7. **Recommendations** - Actionable next steps based on analysis

**Subject for Analysis:** {query}

Provide objective, evidence-based analysis with clear reasoning for all conclusions. Include multiple perspectives where appropriate.
""",

            'comparison': """
You are an expert consultant providing detailed comparison analysis. Structure your response as:

1. **Comparison Overview** - What's being compared and evaluation criteria
2. **Background Context** - Relevant information about each option
3. **Feature-by-Feature Analysis** - Systematic point-by-point comparison
4. **Strengths and Weaknesses** - Detailed pros and cons for each option
5. **Use Case Scenarios** - When each option works best
6. **Cost-Benefit Analysis** - Resource implications and value assessment
7. **Final Recommendation** - Best choice with clear reasoning

**Comparison Request:** {query}

Use tables or structured formats where helpful. Provide balanced, objective analysis while being clear about your final recommendation.
""",

            'creative': """
You are a creative expert and innovative thinker. Approach this creatively while maintaining structure:

1. **Creative Vision** - Innovative concept and inspiration
2. **Conceptual Framework** - Underlying structure and approach
3. **Detailed Development** - Rich exploration of ideas and possibilities  
4. **Creative Examples** - Vivid illustrations and scenarios
5. **Variations and Alternatives** - Different approaches and styles
6. **Implementation Strategy** - How to bring creative ideas to reality
7. **Impact and Outcomes** - Expected results and broader implications

**Creative Challenge:** {query}

Be imaginative and original while providing practical, actionable guidance. Include multiple creative approaches and encourage experimentation.
"""
        }

    def get_template(self, template_type: str, query: str) -> str:
        """Get a formatted template for the query"""
        if template_type in self.templates:
            return self.templates[template_type].format(query=query)
        else:
            return self.templates['explanation'].format(query=query)


# Integration helper functions
def auto_enhance_prompt(prompt: str, **kwargs) -> str:
    """Auto-enhance a prompt with smart defaults"""
    enhancer = PromptEnhancer()

    # Auto-detect best format based on prompt content
    prompt_lower = prompt.lower()

    if any(word in prompt_lower for word in ['how to', 'steps', 'process', 'guide']):
        format_type = ResponseFormat.STEP_BY_STEP
    elif any(word in prompt_lower for word in ['compare', 'vs', 'versus', 'difference']):
        format_type = ResponseFormat.COMPARATIVE
    elif any(word in prompt_lower for word in ['analyze', 'analysis', 'evaluate']):
        format_type = ResponseFormat.ANALYTICAL
    elif any(word in prompt_lower for word in ['create', 'design', 'write', 'compose']):
        format_type = ResponseFormat.CREATIVE
    elif any(word in prompt_lower for word in ['research', 'study', 'investigate']):
        format_type = ResponseFormat.RESEARCH
    elif any(word in prompt_lower for word in ['problem', 'solve', 'issue', 'challenge']):
        format_type = ResponseFormat.PROBLEM_SOLVING
    elif any(word in prompt_lower for word in ['technical', 'implement', 'code', 'system']):
        format_type = ResponseFormat.TECHNICAL
    else:
        format_type = ResponseFormat.DETAILED

    return enhancer.enhance_prompt(prompt, format_type, **kwargs)


def get_template_prompt(prompt: str, template_type: str = None) -> str:
    """Get a template-based enhanced prompt"""
    template = PromptTemplate()

    # Auto-detect template type if not specified
    if not template_type:
        prompt_lower = prompt.lower()
        if any(word in prompt_lower for word in ['how to', 'steps', 'guide']):
            template_type = 'how_to'
        elif any(word in prompt_lower for word in ['compare', 'vs', 'versus']):
            template_type = 'comparison'
        elif any(word in prompt_lower for word in ['analyze', 'analysis']):
            template_type = 'analysis'
        elif any(word in prompt_lower for word in ['create', 'design', 'write']):
            template_type = 'creative'
        else:
            template_type = 'explanation'

    return template.get_template(template_type, prompt)
