# utils/quick_enhance.py
from typing import Optional, Dict, Any
import re


def quick_enhance(prompt: str, style: str = "detailed") -> str:
    """Quick prompt enhancement with predefined styles"""

    enhancers = {
        "detailed": _enhance_detailed,
        "analytical": _enhance_analytical,
        "howto": _enhance_howto,
        "comparison": _enhance_comparison,
        "creative": _enhance_creative,
        "research": _enhance_research,
        "coding": _enhance_coding,
        "explanation": _enhance_explanation
    }

    enhancer = enhancers.get(style, enhancers["detailed"])
    return enhancer(prompt)


def _enhance_detailed(prompt: str) -> str:
    """Enhance for detailed, comprehensive responses"""
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


def _enhance_analytical(prompt: str) -> str:
    """Enhance for analytical responses"""
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


def _enhance_howto(prompt: str) -> str:
    """Enhance for how-to and instructional responses"""
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


def _enhance_comparison(prompt: str) -> str:
    """Enhance for comparative analysis"""
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


def _enhance_creative(prompt: str) -> str:
    """Enhance for creative and innovative responses"""
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


def _enhance_research(prompt: str) -> str:
    """Enhance for research-oriented responses"""
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


def _enhance_coding(prompt: str) -> str:
    """Enhance for coding and programming tasks"""
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


def _enhance_explanation(prompt: str) -> str:
    """Enhance for educational explanations"""
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


def smart_enhance(prompt: str) -> str:
    """Automatically detect the best enhancement style based on prompt content"""
    prompt_lower = prompt.lower()

    # Pattern matching for different types of queries
    if any(word in prompt_lower for word in ['how to', 'steps', 'process', 'guide', 'tutorial']):
        return quick_enhance(prompt, "howto")
    elif any(word in prompt_lower for word in ['compare', 'vs', 'versus', 'difference', 'better']):
        return quick_enhance(prompt, "comparison")
    elif any(word in prompt_lower for word in ['analyze', 'analysis', 'evaluate', 'assess', 'examine']):
        return quick_enhance(prompt, "analytical")
    elif any(word in prompt_lower for word in ['create', 'design', 'write', 'compose', 'generate']):
        return quick_enhance(prompt, "creative")
    elif any(word in prompt_lower for word in ['research', 'study', 'investigate', 'find out', 'sources']):
        return quick_enhance(prompt, "research")
    elif any(word in prompt_lower for word in ['code', 'program', 'implement', 'algorithm', 'function']):
        return quick_enhance(prompt, "coding")
    elif any(word in prompt_lower for word in ['explain', 'what is', 'definition', 'meaning', 'understand']):
        return quick_enhance(prompt, "explanation")
    else:
        return quick_enhance(prompt, "detailed")


def enhance_with_context(prompt: str,
                         context: str = None,
                         audience: str = None,
                         purpose: str = None,
                         style: str = "detailed") -> str:
    """Enhance prompt with additional context information"""

    base_enhanced = quick_enhance(prompt, style)

    # Add context information
    context_additions = []

    if context:
        context_additions.append(f"**Additional Context:** {context}")

    if audience:
        context_additions.append(f"**Target Audience:** {audience} - Adjust complexity and examples accordingly.")

    if purpose:
        context_additions.append(f"**Purpose:** {purpose} - Focus your response to achieve this goal.")

    if context_additions:
        context_section = "\n".join(context_additions)
        return f"{base_enhanced}\n\n{context_section}"

    return base_enhanced


# Utility functions for common enhancement patterns
def enhance_for_beginners(prompt: str) -> str:
    """Enhance prompt for beginner-friendly responses"""
    return enhance_with_context(
        prompt,
        audience="beginners with no prior knowledge",
        purpose="education and understanding",
        style="explanation"
    )


def enhance_for_professionals(prompt: str) -> str:
    """Enhance prompt for professional/expert responses"""
    return enhance_with_context(
        prompt,
        audience="professionals and experts in the field",
        purpose="advanced analysis and implementation",
        style="analytical"
    )


def enhance_for_decision_making(prompt: str) -> str:
    """Enhance prompt to support decision making"""
    return enhance_with_context(
        prompt,
        purpose="support informed decision-making",
        style="comparison"
    )


# Example usage and testing
if __name__ == "__main__":
    # Test different enhancement styles
    test_prompt = "What is machine learning?"

    print("Original prompt:", test_prompt)
    print("\n" + "=" * 80)

    styles = ["detailed", "analytical", "explanation", "research"]

    for style in styles:
        print(f"\n=== {style.upper()} ENHANCEMENT ===")
        enhanced = quick_enhance(test_prompt, style)
        print(enhanced[:300] + "..." if len(enhanced) > 300 else enhanced)

    print("\n" + "=" * 80)
    print("=== SMART ENHANCEMENT ===")
    smart_enhanced = smart_enhance("How to implement a neural network?")
    print(smart_enhanced[:300] + "..." if len(smart_enhanced) > 300 else smart_enhanced)
