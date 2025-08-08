# graphs/reasoning_graph.py
from langgraph.graph import  END, START, StateGraph
from pydantic import BaseModel
from halting.energy_monitor import EnergyMonitor

# Define the state schema using Pydantic
class ReasoningState(BaseModel):
    context: str  # Current context or input
    loop : bool  # Whether to continue the loop
    step_count: int = 0  # ✅ Add step counter with default value

class InputState(BaseModel):
    context: str
    loop : bool  # Whether to continue the loop
    step_count: int = 0  # ✅ Add step counter with default value

class OutputState(BaseModel):
    context: str
    loop : bool  # Whether to continue the loop
    step_count: int = 0  # ✅ Add step counter with default value


def build_reasoning_graph(chain, memory_manager, logger):
    INITIAL_STEP_COUNT = 1
    energy_monitor = EnergyMonitor()
    graph = StateGraph(state_schema=ReasoningState, input_schema=InputState)

    def get_augmented_context(state: ReasoningState) -> str:
        retrieved_docs = memory_manager.vector_store.as_retriever().get_relevant_documents(state.context)

        augmented_context = state.context
        for doc in retrieved_docs:
            augmented_context += "\n\n" + doc.page_content
        return augmented_context

    def process_chain_output(context: str, output: str) -> None:
        logger.log_step(context, output)
        memory_manager.save(context, output)

    def reasoning_node(state: ReasoningState) -> ReasoningState:
        augmented_context = get_augmented_context(state)
        output = chain.run(context=augmented_context)

        process_chain_output(state.context, output)
        should_continue = energy_monitor.should_continue(output)

        return ReasoningState(
            context=output,
            loop=should_continue,
            step_count=state.step_count + 1
        )

    def start_node(state: InputState) -> ReasoningState:
        return ReasoningState(
            context=state.context,
            loop=True,
            step_count=INITIAL_STEP_COUNT
        )

    # Graph structure setup
    start_node_name = "start"
    reasoning_node_name = "reasoning"

    # Add nodes with proper functions
    graph.add_node(start_node_name, start_node)
    graph.add_node(reasoning_node_name, reasoning_node)

    # Edge configuration
    graph.add_edge(start_node_name, reasoning_node_name)
    graph.set_entry_point(start_node_name)

    graph.add_conditional_edges(
        reasoning_node_name,
        lambda state: reasoning_node_name if state.loop else "END"
    )

    return graph.compile()

def build_reasoning_graph_bak(chain, memory_manager, logger):
    energy_monitor = EnergyMonitor()

    graph = StateGraph(state_schema=ReasoningState, input_schema=InputState)

    def reasoning_node(state: ReasoningState) -> ReasoningState:
        # Retrieve relevant documents from the vector store
        try:
            retrieved_docs = memory_manager.retriever.get_relevant_documents(state.context)
        except AttributeError:
            # If 'retriever' is not the correct attribute name, try '_retriever' or inspect the object
            retrieved_docs = memory_manager._retriever.get_relevant_documents(state.context)

        # retrieved_docs = memory_manager.retriever.get_relevant_documents(state.context)

        # Create augmented context by combining retrieved documents with the current context
        augmented_context = state.context
        for doc in retrieved_docs:
            augmented_context += "\n\n" + doc.page_content  # Combine with retrieved documents

        # Run the chain with the augmented context
        output = chain.run(context=augmented_context)

        # Log the step and save to memory
        logger.log_step(state.context, output)
        memory_manager.save(state.context, output)

        # Determine if the graph should continue
        should_continue = energy_monitor.should_continue(output)

        # Return the updated state
        return ReasoningState(context=output,
                              loop=should_continue,
                              step_count=state.step_count + 1 )

        # Initialize the StateGraph with the defined schema
        # graph = StateGraph(state_schema=ReasoningState)

        graph.add_node("START", InputState(context="", loop=True, step_count=1))
        # Add the node and edges
        graph.add_node("reasoning_node" + step_count, reasoning_node)
        graph.add_edge(START, "START")

        graph.add_edge("reasoning_edge" + step_count, "reasoning_node")
        graph.set_entry_point("reasoning_node")  # ✅ Add this line to define the entry point

        graph.add_conditional_edges(
            "reasoning_node",
            lambda state: "reasoning_node" if state.loop else END
    )

    return graph.compile()
