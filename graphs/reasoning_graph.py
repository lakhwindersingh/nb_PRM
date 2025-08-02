# graphs/reasoning_graph.py
from langgraph.graph import END, StateGraph
from halting.energy_monitor import EnergyMonitor

def build_reasoning_graph(chain, memory_manager, logger):
    energy_monitor = EnergyMonitor()

    def thinking_node(state):
        context = state.get("context", "")
        augmented_context = memory_manager.get_augmented_context(context)
        output = chain.run(context=augmented_context)

        logger.log_step(context, output)
        memory_manager.save(context, output)

        should_continue = energy_monitor.should_continue(output)
        return {"context": output, "continue": should_continue}

    graph = StateGraph()
    graph.add_node("think", thinking_node)
    graph.set_entry_point("think")
    graph.add_conditional_edges(
        "think",
        lambda state: "think" if state["continue"] else END
    )

    return graph.compile()
