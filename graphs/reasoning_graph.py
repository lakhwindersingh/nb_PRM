# graphs/reasoning_graph.py
from langgraph.graph import END, StateGraph

def build_reasoning_graph(chain, memory_manager, logger):
    def thinking_node(state):
        context = state.get("context", "")
        output = chain.run(context=context)
        logger.log_step(context, output)
        memory_manager.save(context, output)
        return {"context": output}

    graph = StateGraph()
    graph.add_node("think", thinking_node)
    graph.set_entry_point("think")
    graph.add_edge("think", "think")  # perpetual loop
    graph.add_conditional_edges("think", lambda _: END)  # Replace with stop logic if needed

    return graph.compile()
