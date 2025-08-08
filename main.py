# main.py
import argparse

# from chains.reasoning_chain import create_reasoning_chain
from chains.reasoning_chain_google import create_reasoning_chain
from graphs.reasoning_graph import build_reasoning_graph
from logger.step_logger import StepLogger
from memory.memory_manager import MemoryManager

# CLI Setup
parser = argparse.ArgumentParser(description="Run the Perpetual Reasoning Model")
parser.add_argument("--context", type=str, help="Initial reasoning context")
parser.add_argument("--steps", type=int, default=10, help="Maximum number of reasoning steps")
args = parser.parse_args()


# Setup
memory = MemoryManager()
logger = StepLogger("data/logs.jsonl")
chain = create_reasoning_chain(memory.retriever_memory)
reasoning_graph = build_reasoning_graph(chain, memory, logger)

# Run perpetual thinking loop
print("\n[ Perpetual Reasoning Model Started ]\n")

# Starting context
initial_state = {
    "context": "Why do humans seek meaning in life?",
    "loop": True,
    "step_count": 0
}
context = args.context

result = reasoning_graph.invoke(initial_state)
print(result)  # Should return a valid state with updated step_count


# for _ in range(args.steps):
#     result = reasoning_graph.invoke({"context": initial_context, "contin": True, "step_count": 0 })
#     context = result.get("context")
#     if context.strip().upper() == "STOP":
#         print("\n[ Reasoning completed. STOP received. ]")
#         break