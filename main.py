# main.py
from chains.reasoning_chain import create_reasoning_chain
from memory.memory_manager import MemoryManager
from logger.step_logger import StepLogger
from graphs.reasoning_graph import build_reasoning_graph

# Setup
memory = MemoryManager()
logger = StepLogger("data/logs.jsonl")
chain = create_reasoning_chain(memory.retriever)
reasoning_graph = build_reasoning_graph(chain, memory, logger)

# Starting context
initial_context = "Why do humans seek meaning in life?"

# Run perpetual thinking loop
print("\n[ Perpetual Reasoning Model Started ]\n")
reasoning_graph.invoke({"context": initial_context})