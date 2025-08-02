# memory/memory_manager.py
from langchain.memory import VectorStoreRetrieverMemory
from memory.vector_store import get_vector_store

class MemoryManager:
    def __init__(self):
        self.vector_store = get_vector_store()
        self.retriever = VectorStoreRetrieverMemory(retriever=self.vector_store.as_retriever())

    def save(self, context, output):
        self.retriever.save_context({"input": context}, {"output": output})

