# graphs/reasoning_graph.py
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel
from halting.energy_monitor import EnergyMonitor

# Define the state schema using Pydantic
class ReasoningState(BaseModel):
    context: str  # Current context or input
    contin: bool  # Whether to continue the loop

def build_reasoning_graph(chain, memory_manager, logger):
    energy_monitor = EnergyMonitor()
    graph = StateGraph(state_schema=ReasoningState)

    def thinking_node(state: ReasoningState) -> ReasoningState:
        # Get augmented context from memory
        augmented_context = memory_manager.get_augmented_context(state.context)

        # Run the chain with the augmented context
        output = chain.run(context=augmented_context)

        # Log the step and save to memory
        logger.log_step(state.context, output)
        memory_manager.save(state.context, output)

        # Determine if the graph should continue
        should_continue = energy_monitor.should_continue(output)

        # Return the updated state
        return ReasoningState(context=output, contin=should_continue)

    # Add the node
    graph.add_node("think", thinking_node)

    # Add the entry point edge from START to think node
    graph.add_edge(START, "think")

    # Add conditional edges to loop back to think or end
    graph.add_conditional_edges(
        "think",
        lambda state: "think" if state.contin else END,
        {
            "think": "think",
            END: END
        }
    )

    return graph.compile()