# graphs/reasoning_graph.py
from typing import Dict, Any, List, Optional, Callable, Tuple
from enum import Enum
from dataclasses import dataclass
import logging
from datetime import datetime

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from halting.energy_monitor import EnergyMonitor
from memory.memory_manager import EnhancedMemoryManager, ResponseStyle

logger = logging.getLogger(__name__)


class ReasoningMode(Enum):
    """Different reasoning modes for different types of problems"""
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    PROBLEM_SOLVING = "problem_solving"
    RESEARCH = "research"
    STEP_BY_STEP = "step_by_step"
    COMPARATIVE = "comparative"
    AUTO = "auto"


class StepType(Enum):
    """Types of reasoning steps"""
    INITIAL = "initial"
    REASONING = "reasoning"
    REFLECTION = "reflection"
    VALIDATION = "validation"
    SYNTHESIS = "synthesis"
    FINAL = "final"


@dataclass
class ReasoningStep:
    """Individual reasoning step with metadata"""
    step_number: int
    step_type: StepType
    input_context: str
    output: str
    reasoning_mode: ReasoningMode
    energy_score: float
    timestamp: str
    duration_ms: int = 0
    tokens_used: int = 0
    memory_retrieval_count: int = 0


class ReasoningState(BaseModel):
    """Enhanced reasoning state with comprehensive tracking"""
    context: str = Field(description="Current reasoning context")
    original_query: str = Field(description="Original user query")
    loop: bool = Field(default=True, description="Whether to continue reasoning")
    step_count: int = Field(default=0, description="Current step number")
    max_steps: int = Field(default=10, description="Maximum allowed steps")
    reasoning_mode: ReasoningMode = Field(default=ReasoningMode.AUTO, description="Current reasoning mode")

    # Enhanced tracking
    steps_history: List[Dict[str, Any]] = Field(default_factory=list, description="History of reasoning steps")
    accumulated_insights: List[str] = Field(default_factory=list, description="Key insights gathered")
    confidence_scores: List[float] = Field(default_factory=list, description="Confidence scores per step")
    energy_scores: List[float] = Field(default_factory=list, description="Energy scores per step")

    # Metadata
    start_time: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Reasoning start time")
    total_tokens: int = Field(default=0, description="Total tokens used")
    error_count: int = Field(default=0, description="Number of errors encountered")

    class Config:
        arbitrary_types_allowed = True


class InputState(BaseModel):
    """Input state for reasoning graph"""
    context: str = Field(description="Initial context or query")
    reasoning_mode: ReasoningMode = Field(default=ReasoningMode.AUTO, description="Reasoning mode to use")
    max_steps: int = Field(default=10, description="Maximum reasoning steps")
    enable_reflection: bool = Field(default=True, description="Enable reflection steps")
    enable_validation: bool = Field(default=True, description="Enable validation steps")
    custom_prompts: Optional[Dict[str, str]] = Field(default=None, description="Custom prompts for different steps")
    error_count: int = Field(default=0, description="Number of errors encountered")


class OutputState(BaseModel):
    """Output state from reasoning graph"""
    final_output: str = Field(description="Final reasoning output")
    reasoning_trace: List[Dict[str, Any]] = Field(description="Complete reasoning trace")
    total_steps: int = Field(description="Total steps taken")
    success: bool = Field(description="Whether reasoning completed successfully")
    insights: List[str] = Field(description="Key insights discovered")
    confidence_score: float = Field(description="Final confidence score")
    metadata: Dict[str, Any] = Field(description="Additional metadata")
    error_count: int = Field(default=0, description="Number of errors encountered")


class EnhancedReasoningGraph:
    """Enhanced reasoning graph with multiple reasoning strategies and monitoring"""

    def __init__(self,
                 chain: Any,
                 memory_manager: EnhancedMemoryManager,
                 energy_monitor: Optional[EnergyMonitor] = None,
                 reasoning_logger: Optional[Any] = None,
                 enable_reflection: bool = True,
                 enable_validation: bool = True,
                 enable_synthesis: bool = True):

        self.chain = chain
        self.memory_manager = memory_manager
        self.energy_monitor = energy_monitor or EnergyMonitor(threshold=0.7, decay_rate=0.03)
        self.reasoning_logger = reasoning_logger
        self.enable_reflection = enable_reflection
        self.enable_validation = enable_validation
        self.enable_synthesis = enable_synthesis

        # Performance tracking
        self.execution_stats = {
            'total_executions': 0,
            'successful_completions': 0,
            'energy_halts': 0,
            'max_step_halts': 0,
            'error_halts': 0,
            'average_steps': 0.0,
            'reasoning_modes_used': {}
        }

        # Build the graph
        self.compiled_graph = self._build_graph()

        if reasoning_logger:
            reasoning_logger.info("Enhanced Reasoning Graph initialized")
        else:
            logger.info("Enhanced Reasoning Graph initialized")

    def _build_graph(self) -> StateGraph:
        """Build the enhanced reasoning graph"""
        graph = StateGraph(state_schema=ReasoningState, input_schema=InputState)

        # Add nodes
        graph.add_node("start", self._start_node)
        graph.add_node("reasoning", self._reasoning_node)
        graph.add_node("reflection", self._reflection_node)
        graph.add_node("validation", self._validation_node)
        graph.add_node("synthesis", self._synthesis_node)
        graph.add_node("finalize", self._finalize_node)

        # Add edges
        graph.add_edge("start", "reasoning")
        graph.set_entry_point("start")

        # Conditional edges for reasoning flow
        graph.add_conditional_edges(
            "reasoning",
            self._should_continue_reasoning,
            {
                "continue": "reasoning",
                "reflect": "reflection",
                "validate": "validation",
                "synthesize": "synthesis",
                "end": "finalize"
            }
        )

        # Reflection back to reasoning
        if self.enable_reflection:
            graph.add_conditional_edges(
                "reflection",
                self._after_reflection_decision,
                {
                    "continue": "reasoning",
                    "validate": "validation",
                    "synthesize": "synthesis",
                    "end": "finalize"
                }
            )

        # Validation flow
        if self.enable_validation:
            graph.add_conditional_edges(
                "validation",
                self._after_validation_decision,
                {
                    "continue": "reasoning",
                    "synthesize": "synthesis",
                    "end": "finalize"
                }
            )

        # Synthesis flow
        if self.enable_synthesis:
            graph.add_conditional_edges(
                "synthesis",
                self._after_synthesis_decision,
                {
                    "continue": "reasoning",
                    "end": "finalize"
                }
            )

        # Finalize always ends
        graph.add_edge("finalize", END)

        return graph.compile()

    # Interface methods that delegate to the compiled graph
    # graphs/reasoning_graph.py - Update the invoke method

    # graphs/reasoning_graph.py - Fix the invoke method and state handling

    def invoke(self, input_data: Any, config: Optional[Dict] = None) -> ReasoningState:
        """
        Invoke the reasoning graph with input data

        Args:
            input_data: Can be InputState, dict, or legacy ReasoningState
            config: Optional configuration for the graph execution

        Returns:
            Final ReasoningState after processing
        """
        try:
            # Set default config with recursion limit
            if config is None:
                config = {}

            # Set a reasonable recursion limit
            if 'recursion_limit' not in config:
                config['recursion_limit'] = 15

            # Handle different input types for backward compatibility
            if isinstance(input_data, dict):
                if 'context' in input_data:
                    input_state = InputState(**input_data)
                else:
                    input_state = InputState(
                        context=input_data.get('context', ''),
                        reasoning_mode=ReasoningMode(input_data.get('reasoning_mode', 'auto')),
                        max_steps=input_data.get('max_steps', 8)
                    )
            elif isinstance(input_data, ReasoningState):
                input_state = InputState(
                    context=input_data.context,
                    reasoning_mode=input_data.reasoning_mode,
                    max_steps=min(input_data.max_steps, 8)
                )
            else:
                input_state = input_data

            # Ensure max_steps is reasonable
            if input_state.max_steps > 10:
                input_state.max_steps = 10

            # Execute the graph with config
            result = self.compiled_graph.invoke(input_state, config)

            # Handle different result types
            if isinstance(result, dict):
                # Convert dict result to ReasoningState
                reasoning_state = ReasoningState(
                    context=result.get('context', 'No output generated'),
                    original_query=result.get('original_query', input_state.context),
                    loop=result.get('loop', False),
                    step_count=result.get('step_count', 0),
                    max_steps=result.get('max_steps', input_state.max_steps),
                    reasoning_mode=ReasoningMode(result.get('reasoning_mode', 'auto')),
                    steps_history=result.get('steps_history', []),
                    accumulated_insights=result.get('accumulated_insights', []),
                    confidence_scores=result.get('confidence_scores', []),
                    energy_scores=result.get('energy_scores', []),
                    start_time=result.get('start_time', datetime.now().isoformat()),
                    total_tokens=result.get('total_tokens', 0),
                    error_count=result.get('error_count', 0)
                )
                result = reasoning_state
            elif not isinstance(result, ReasoningState):
                # If result is not a ReasoningState, create one
                reasoning_state = ReasoningState(
                    context=str(result),
                    original_query=input_state.context,
                    loop=False,
                    step_count=0,
                    max_steps=input_state.max_steps,
                    reasoning_mode=input_state.reasoning_mode,
                    error_count=0
                )
                result = reasoning_state

            # Update execution stats
            self.execution_stats['total_executions'] += 1
            if hasattr(result, 'error_count') and result.error_count == 0:
                self.execution_stats['successful_completions'] += 1

            return result

        except Exception as e:
            self.execution_stats['total_executions'] += 1
            self.execution_stats['error_halts'] += 1

            if self.reasoning_logger:
                self.reasoning_logger.error(f"Graph execution failed: {e}")
            else:
                logger.error(f"Graph execution failed: {e}")

            # Return error state
            error_state = ReasoningState(
                context=f"Error during reasoning: {str(e)}",
                original_query=getattr(input_data, 'context', str(input_data)[:100]),
                loop=False,
                error_count=1,
                step_count=0,
                max_steps=getattr(input_data, 'max_steps', 10),
                reasoning_mode=ReasoningMode.AUTO
            )
            return error_state

    def stream(self, input_data: Any, config: Optional[Dict] = None):
        """Stream the reasoning process step by step"""
        try:
            # Handle input conversion similar to invoke
            if isinstance(input_data, dict):
                if 'context' in input_data:
                    input_state = InputState(**input_data)
                else:
                    input_state = InputState(
                        context=input_data.get('context', ''),
                        reasoning_mode=ReasoningMode(input_data.get('reasoning_mode', 'auto')),
                        max_steps=input_data.get('max_steps', 10)
                    )
            elif isinstance(input_data, ReasoningState):
                input_state = InputState(
                    context=input_data.context,
                    reasoning_mode=input_data.reasoning_mode,
                    max_steps=input_data.max_steps
                )
            else:
                input_state = input_data

            # Stream the graph execution
            for step in self.compiled_graph.stream(input_state, config):
                yield step

        except Exception as e:
            if self.reasoning_logger:
                self.reasoning_logger.error(f"Graph streaming failed: {e}")
            else:
                logger.error(f"Graph streaming failed: {e}")

            # Yield error state
            yield {"error": str(e)}

    def get_graph(self):
        """Get the compiled graph for direct access"""
        return self.compiled_graph

    def execute(self, input_state: InputState) -> OutputState:
        """Execute the reasoning graph and return comprehensive output"""
        try:
            # Run the graph
            result_state = self.invoke(input_state)

            # Create output state
            output = OutputState(
                final_output=result_state.context,
                reasoning_trace=result_state.steps_history,
                total_steps=result_state.step_count,
                success=result_state.error_count == 0,
                insights=result_state.accumulated_insights,
                confidence_score=result_state.confidence_scores[-1] if result_state.confidence_scores else 0.5,
                metadata={
                    'reasoning_mode': result_state.reasoning_mode.value,
                    'energy_scores': result_state.energy_scores,
                    'start_time': result_state.start_time,
                    'error_count': result_state.error_count,
                    'original_query': result_state.original_query
                }
            )

            return output

        except Exception as e:
            if self.reasoning_logger:
                self.reasoning_logger.error(f"Graph execution failed: {e}")
            else:
                logger.error(f"Graph execution failed: {e}")

            # Return error output
            return OutputState(
                final_output=f"Reasoning failed: {str(e)}",
                reasoning_trace=[],
                total_steps=0,
                success=False,
                insights=[],
                confidence_score=0.0,
                metadata={'error': str(e)}
            )

    # graphs/reasoning_graph.py - Fix all node methods to properly handle state

    def _start_node(self, state: InputState) -> ReasoningState:
        """Initialize reasoning state from input"""
        try:
            # Detect reasoning mode if auto
            reasoning_mode = state.reasoning_mode
            if reasoning_mode == ReasoningMode.AUTO:
                reasoning_mode = self._detect_reasoning_mode(state.context)

            # Track mode usage
            mode_name = reasoning_mode.value
            self.execution_stats['reasoning_modes_used'][mode_name] = \
                self.execution_stats['reasoning_modes_used'].get(mode_name, 0) + 1

            return ReasoningState(
                context=state.context,
                original_query=state.context,
                reasoning_mode=reasoning_mode,
                max_steps=state.max_steps,
                step_count=0,
                loop=True,
                steps_history=[],
                accumulated_insights=[],
                confidence_scores=[],
                energy_scores=[],
                start_time=datetime.now().isoformat(),
                total_tokens=0,
                error_count=0
            )

        except Exception as e:
            logger.error(f"Error in start node: {e}")
            # Return minimal error state
            return ReasoningState(
                context=getattr(state, 'context', 'Error in initialization'),
                original_query=getattr(state, 'context', 'Unknown'),
                loop=False,
                step_count=0,
                max_steps=getattr(state, 'max_steps', 10),
                reasoning_mode=ReasoningMode.AUTO,
                error_count=1,
                steps_history=[],
                accumulated_insights=[],
                confidence_scores=[],
                energy_scores=[],
                start_time=datetime.now().isoformat(),
                total_tokens=0
            )

    def _reasoning_node(self, state: ReasoningState) -> ReasoningState:
        """Main reasoning node with enhanced context and monitoring"""
        start_time = datetime.now()

        try:
            # Ensure we have a proper ReasoningState
            if not isinstance(state, ReasoningState):
                logger.error(f"Expected ReasoningState, got {type(state)}")
                return ReasoningState(
                    context="Invalid state type in reasoning node",
                    original_query="Unknown",
                    loop=False,
                    error_count=1
                )

            # Get enhanced context based on reasoning mode
            enhanced_prompt = self._get_enhanced_reasoning_prompt(state)

            # Get augmented context from memory
            try:
                retrieved_context = self.memory_manager.get_augmented_context(
                    state.original_query,
                    k=6,
                    context_length_limit=3000
                )
            except Exception as e:
                logger.warning(f"Memory retrieval failed: {e}")
                retrieved_context = "Context retrieval failed."

            # Combine enhanced prompt with retrieved context
            full_context = f"{enhanced_prompt}\n\nRelevant Context:\n{retrieved_context}"

            # Run the chain
            output = self._run_chain_safely(full_context)

            # Calculate metrics
            duration = int((datetime.now() - start_time).total_seconds() * 1000)
            energy_score = self.energy_monitor.compute_energy(output)

            # Create reasoning step record
            step = ReasoningStep(
                step_number=state.step_count + 1,
                step_type=StepType.REASONING,
                input_context=state.context[:500],
                output=output,
                reasoning_mode=state.reasoning_mode,
                energy_score=energy_score,
                timestamp=start_time.isoformat(),
                duration_ms=duration
            )

            # Create new state (don't modify the original)
            new_state = ReasoningState(
                context=output,
                original_query=state.original_query,
                loop=state.loop,
                step_count=state.step_count + 1,
                max_steps=state.max_steps,
                reasoning_mode=state.reasoning_mode,
                steps_history=state.steps_history + [step.__dict__],
                accumulated_insights=state.accumulated_insights.copy(),
                confidence_scores=state.confidence_scores.copy(),
                energy_scores=state.energy_scores + [energy_score],
                start_time=state.start_time,
                total_tokens=state.total_tokens,
                error_count=state.error_count
            )

            # Log step
            if self.reasoning_logger:
                self.reasoning_logger.log_step(state.context, output)

            # Save to memory
            try:
                self.memory_manager.save_interaction(state.context, output, {
                    'step_type': 'reasoning',
                    'step_number': new_state.step_count,
                    'reasoning_mode': state.reasoning_mode.value,
                    'energy_score': energy_score
                })
            except Exception as e:
                logger.warning(f"Failed to save interaction: {e}")

            # Extract insights
            insights = self._extract_insights(output)
            new_state.accumulated_insights.extend(insights)

            return new_state

        except Exception as e:
            logger.error(f"Error in reasoning node: {e}")
            # Create error state
            return ReasoningState(
                context=f"Error in reasoning: {str(e)}",
                original_query=getattr(state, 'original_query', 'Unknown'),
                loop=False,
                step_count=getattr(state, 'step_count', 0),
                max_steps=getattr(state, 'max_steps', 10),
                reasoning_mode=getattr(state, 'reasoning_mode', ReasoningMode.AUTO),
                steps_history=getattr(state, 'steps_history', []),
                accumulated_insights=getattr(state, 'accumulated_insights', []),
                confidence_scores=getattr(state, 'confidence_scores', []),
                energy_scores=getattr(state, 'energy_scores', []),
                start_time=getattr(state, 'start_time', datetime.now().isoformat()),
                total_tokens=getattr(state, 'total_tokens', 0),
                error_count=getattr(state, 'error_count', 0) + 1
            )

    def _finalize_node(self, state: ReasoningState) -> ReasoningState:
        """Finalize reasoning and prepare output"""
        try:
            # Ensure we have a proper ReasoningState
            if not isinstance(state, ReasoningState):
                logger.error(f"Expected ReasoningState in finalize, got {type(state)}")
                return ReasoningState(
                    context="Invalid state in finalize node",
                    original_query="Unknown",
                    loop=False,
                    error_count=1
                )

            # Update execution stats
            if hasattr(self, 'execution_stats'):
                self.execution_stats['total_executions'] = self.execution_stats.get('total_executions', 0) + 1

                if state.error_count == 0:
                    self.execution_stats['successful_completions'] = self.execution_stats.get('successful_completions',
                                                                                              0) + 1

                # Update average steps
                total_steps = len(state.steps_history)
                current_avg = self.execution_stats.get('average_steps', 0.0)
                total_executions = self.execution_stats['total_executions']
                if total_executions > 0:
                    self.execution_stats['average_steps'] = (current_avg * (
                                total_executions - 1) + total_steps) / total_executions

            # Calculate final confidence score
            final_confidence = 0.5
            if state.energy_scores:
                final_confidence = sum(state.energy_scores[-3:]) / min(3, len(state.energy_scores))

            # Create final state
            final_state = ReasoningState(
                context=state.context,
                original_query=state.original_query,
                loop=False,  # Always stop after finalization
                step_count=state.step_count,
                max_steps=state.max_steps,
                reasoning_mode=state.reasoning_mode,
                steps_history=state.steps_history,
                accumulated_insights=state.accumulated_insights,
                confidence_scores=state.confidence_scores + [final_confidence],
                energy_scores=state.energy_scores,
                start_time=state.start_time,
                total_tokens=state.total_tokens,
                error_count=state.error_count
            )

            return final_state

        except Exception as e:
            logger.error(f"Error in finalize node: {e}")
            return ReasoningState(
                context=getattr(state, 'context', f"Finalization error: {str(e)}"),
                original_query=getattr(state, 'original_query', 'Unknown'),
                loop=False,
                step_count=getattr(state, 'step_count', 0),
                max_steps=getattr(state, 'max_steps', 10),
                reasoning_mode=getattr(state, 'reasoning_mode', ReasoningMode.AUTO),
                steps_history=getattr(state, 'steps_history', []),
                accumulated_insights=getattr(state, 'accumulated_insights', []),
                confidence_scores=getattr(state, 'confidence_scores', []),
                energy_scores=getattr(state, 'energy_scores', []),
                start_time=getattr(state, 'start_time', datetime.now().isoformat()),
                total_tokens=getattr(state, 'total_tokens', 0),
                error_count=getattr(state, 'error_count', 0) + 1
            )

    # Include all the other node methods (_reflection_node, _validation_node, etc.)
    # and helper methods from the previous implementation...

    def _reflection_node(self, state: ReasoningState) -> ReasoningState:
        """Reflection node to analyze and improve reasoning"""
        if not self.enable_reflection:
            return state

        try:
            reflection_prompt = self._create_reflection_prompt(state)
            reflection_output = self._run_chain_safely(reflection_prompt)

            step = ReasoningStep(
                step_number=state.step_count + 1,
                step_type=StepType.REFLECTION,
                input_context=state.context[:500],
                output=reflection_output,
                reasoning_mode=state.reasoning_mode,
                energy_score=self.energy_monitor.compute_energy(reflection_output),
                timestamp=datetime.now().isoformat()
            )

            new_state = state.copy()
            new_state.context = reflection_output
            new_state.step_count += 1
            new_state.steps_history.append(step.__dict__)

            return new_state

        except Exception as e:
            logger.error(f"Error in reflection node: {e}")
            return state

    def _validation_node(self, state: ReasoningState) -> ReasoningState:
        """Validation node to check reasoning quality"""
        if not self.enable_validation:
            return state

        try:
            validation_prompt = self._create_validation_prompt(state)
            validation_output = self._run_chain_safely(validation_prompt)

            step = ReasoningStep(
                step_number=state.step_count + 1,
                step_type=StepType.VALIDATION,
                input_context=state.context[:500],
                output=validation_output,
                reasoning_mode=state.reasoning_mode,
                energy_score=self.energy_monitor.compute_energy(validation_output),
                timestamp=datetime.now().isoformat()
            )

            new_state = state.copy()
            new_state.step_count += 1
            new_state.steps_history.append(step.__dict__)

            # Don't update main context with validation, just add insights
            insights = self._extract_insights(validation_output)
            new_state.accumulated_insights.extend(insights)

            return new_state

        except Exception as e:
            logger.error(f"Error in validation node: {e}")
            return state

    def _synthesis_node(self, state: ReasoningState) -> ReasoningState:
        """Synthesis node to combine insights and create final output"""
        if not self.enable_synthesis:
            return state

        try:
            synthesis_prompt = self._create_synthesis_prompt(state)
            synthesis_output = self._run_chain_safely(synthesis_prompt)

            step = ReasoningStep(
                step_number=state.step_count + 1,
                step_type=StepType.SYNTHESIS,
                input_context=state.context[:500],
                output=synthesis_output,
                reasoning_mode=state.reasoning_mode,
                energy_score=self.energy_monitor.compute_energy(synthesis_output),
                timestamp=datetime.now().isoformat()
            )

            new_state = state.copy()
            new_state.context = synthesis_output
            new_state.step_count += 1
            new_state.steps_history.append(step.__dict__)
            new_state.loop = False  # Synthesis typically ends reasoning

            return new_state

        except Exception as e:
            logger.error(f"Error in synthesis node: {e}")
            return state

    # Decision methods
    # graphs/reasoning_graph.py - Update the decision methods to prevent infinite loops

    def _should_continue_reasoning(self, state: ReasoningState) -> str:
        """Decide whether to continue reasoning or move to next phase"""
        try:
            # Check max steps first
            if state.step_count >= state.max_steps:
                self.execution_stats['max_step_halts'] += 1
                return "end"

            # Check energy/quality - be more strict to avoid infinite loops
            if state.energy_scores:
                if not self.energy_monitor.should_continue(state.context):
                    self.execution_stats['energy_halts'] += 1
                    if self.enable_synthesis and state.step_count > 2:
                        return "synthesize"
                    return "end"

                # Additional check: if last few steps have low energy, stop
                if len(state.energy_scores) >= 3:
                    recent_avg = sum(state.energy_scores[-3:]) / 3
                    if recent_avg < 0.5:  # Low quality threshold
                        return "end"

            # Limit reasoning loops - after 5 steps, start moving to other phases
            if state.step_count >= 5:
                if self.enable_synthesis:
                    return "synthesize"
                return "end"

            # More conservative reflection/validation triggers
            if state.step_count >= 3:
                if self.enable_reflection and state.step_count % 4 == 0:  # Less frequent
                    return "reflect"
                if self.enable_validation and state.step_count % 3 == 0:  # Less frequent
                    return "validate"

            # Continue reasoning only for first few steps
            if state.step_count < 3:
                return "continue"

            # Default to ending after reasonable steps
            return "end"

        except Exception as e:
            logger.error(f"Error in continue decision: {e}")
            return "end"

    def _after_reflection_decision(self, state: ReasoningState) -> str:
        """Decide what to do after reflection - be more decisive"""
        if state.step_count >= state.max_steps - 1:  # Leave room for finalization
            return "end"

        # After reflection, prefer to synthesize or end rather than continue indefinitely
        if state.step_count >= 4:
            if self.enable_synthesis:
                return "synthesize"
            return "end"

        # Only continue for one more step after reflection
        return "continue"

    def _after_validation_decision(self, state: ReasoningState) -> str:
        """Decide what to do after validation - be more decisive"""
        if state.step_count >= state.max_steps - 1:  # Leave room for finalization
            return "end"

        # After validation, prefer to synthesize or end
        if state.step_count >= 3:
            if self.enable_synthesis:
                return "synthesize"
            return "end"

        # Only continue for one more step after validation
        return "continue"

    def _after_synthesis_decision(self, state: ReasoningState) -> str:
        """Decide what to do after synthesis - always end"""
        return "end"

    def _after_reflection_decision(self, state: ReasoningState) -> str:
        """Decide what to do after reflection"""
        if state.step_count >= state.max_steps:
            return "end"
        if self.enable_validation and state.step_count % 4 == 0:
            return "validate"
        if self.enable_synthesis and state.step_count > 5:
            return "synthesize"
        return "continue"

    def _after_validation_decision(self, state: ReasoningState) -> str:
        """Decide what to do after validation"""
        if state.step_count >= state.max_steps:
            return "end"
        if self.enable_synthesis and state.step_count > 4:
            return "synthesize"
        return "continue"

    def _after_synthesis_decision(self, state: ReasoningState) -> str:
        """Decide what to do after synthesis"""
        return "end"

    # Helper methods - include all the prompt generation and utility methods
    # from the previous implementation...

    def _detect_reasoning_mode(self, context: str) -> ReasoningMode:
        """Auto-detect the best reasoning mode for the context"""
        context_lower = context.lower()

        if any(word in context_lower for word in ['analyze', 'analysis', 'examine', 'evaluate']):
            return ReasoningMode.ANALYTICAL
        elif any(word in context_lower for word in ['create', 'design', 'imagine', 'invent']):
            return ReasoningMode.CREATIVE
        elif any(word in context_lower for word in ['problem', 'solve', 'fix', 'resolve', 'issue']):
            return ReasoningMode.PROBLEM_SOLVING
        elif any(word in context_lower for word in ['research', 'study', 'investigate', 'explore']):
            return ReasoningMode.RESEARCH
        elif any(word in context_lower for word in ['steps', 'how to', 'guide', 'process']):
            return ReasoningMode.STEP_BY_STEP
        elif any(word in context_lower for word in ['compare', 'versus', 'vs', 'difference']):
            return ReasoningMode.COMPARATIVE
        else:
            return ReasoningMode.ANALYTICAL

    def _get_enhanced_reasoning_prompt(self, state: ReasoningState) -> str:
        """Get enhanced prompt based on reasoning mode and current state"""
        return f"""
You are conducting {state.reasoning_mode.value.replace('_', ' ')} reasoning.

Original Query: {state.original_query}
Current Step: {state.step_count + 1}
Current Context: {state.context}

Previous Insights: {'; '.join(state.accumulated_insights[-3:]) if state.accumulated_insights else 'None yet'}

Continue reasoning systematically while building on previous insights.
"""

    def _create_reflection_prompt(self, state: ReasoningState) -> str:
        """Create reflection prompt"""
        return f"""
Reflect on the reasoning process so far:

Original Query: {state.original_query}
Current Understanding: {state.context}

What has been learned? What might be missing? How can reasoning be improved?
"""

    def _create_validation_prompt(self, state: ReasoningState) -> str:
        """Create validation prompt"""
        return f"""
Validate the current reasoning:

Query: {state.original_query}
Current Reasoning: {state.context}

Are the conclusions sound? What are potential weaknesses?
"""

    def _create_synthesis_prompt(self, state: ReasoningState) -> str:
        """Create synthesis prompt"""
        return f"""
Synthesize all insights into a comprehensive response:

Original Query: {state.original_query}
Key Insights: {'; '.join(state.accumulated_insights)}

Provide a complete, well-structured answer.
"""

    def _extract_insights(self, output: str) -> List[str]:
        """Extract key insights from output"""
        # Simple implementation - could be enhanced with NLP
        return []

    def _run_chain_safely(self, context: str) -> str:
        """Run the chain with error handling"""
        try:
            if hasattr(self.chain, 'run'):
                return self.chain.run(context=context)
            elif hasattr(self.chain, 'invoke'):
                return self.chain.invoke({'context': context})
            elif callable(self.chain):
                return self.chain(context)
            else:
                raise ValueError("Chain object doesn't have expected interface")

        except Exception as e:
            logger.error(f"Chain execution failed: {e}")
            return f"Error in reasoning: {str(e)}"

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        return dict(self.execution_stats)


# Factory function - updated to work with both new and legacy interfaces
def build_reasoning_graph(chain,
                          memory_manager,
                          reasoning_logger=None,
                          **kwargs) -> EnhancedReasoningGraph:
    """
    Build an enhanced reasoning graph (backward compatible)

    Args:
        chain: The language model chain to use
        memory_manager: Memory manager instance (EnhancedMemoryManager or legacy)
        reasoning_logger: Optional logger for step tracking
        **kwargs: Additional configuration options
    """
    # Handle different memory manager types
    if hasattr(memory_manager, 'get_augmented_context'):
        # It's either EnhancedMemoryManager or legacy MemoryManager
        mem_mgr = memory_manager
    else:
        # Try to wrap it
        from memory.memory_manager import EnhancedMemoryManager
        if isinstance(memory_manager, EnhancedMemoryManager):
            mem_mgr = memory_manager
        else:
            # Create a wrapper or use as-is
            mem_mgr = memory_manager

    return EnhancedReasoningGraph(
        chain=chain,
        memory_manager=mem_mgr,
        reasoning_logger=reasoning_logger,
        **kwargs
    )


# Usage example
if __name__ == "__main__":
    from memory.memory_manager import create_enhanced_memory_manager


    # Mock chain for testing
    class MockChain:
        def run(self, context):
            return f"Reasoning step response to: {context[:100]}..."


    # Create components
    memory_manager = create_enhanced_memory_manager()
    mock_chain = MockChain()

    # Build reasoning graph
    reasoning_graph = build_reasoning_graph(
        chain=mock_chain,
        memory_manager=memory_manager
    )

    # Test execution - both interfaces work
    print("Testing invoke interface...")
    result = reasoning_graph.invoke({
        "context": "How can we solve climate change?",
        "reasoning_mode": "problem_solving",
        "max_steps": 5
    })

    print(f"Final context: {result.context[:200]}...")
    print(f"Steps taken: {result.step_count}")
    print(f"Success: {result.error_count == 0}")

    print("Enhanced Reasoning Graph test completed!")
