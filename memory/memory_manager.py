# memory/memory_manager.py
from langchain.memory import VectorStoreRetrieverMemory

from memory.vector_store import get_vector_store
# memory/memory_manager.py
from langchain.memory import VectorStoreRetrieverMemory
from memory.vector_store import get_vector_store
from rag.external_retriever import get_external_vector_store
from rag.composite_memory import CompositeRetriever

EXTERNAL_URLS = [
    "https://en.wikipedia.org/wiki/Meaning_of_life",
    "https://plato.stanford.edu/entries/consciousness/"
]

class MemoryManager:
    def __init__(self):
        self.vector_store = get_vector_store()
        self.retriever_memory = VectorStoreRetrieverMemory(retriever=self.vector_store.as_retriever())
        self.external_vector_store = get_external_vector_store(EXTERNAL_URLS)

        self.composite_retriever = CompositeRetriever(
            internal_retriever=self.vector_store.as_retriever(),
            external_vector_store=self.external_vector_store
        )

    def get_augmented_context(self, query):
        return self.composite_retriever.get_combined_context(query)

    def save(self, context, output):
        self.retriever_memory.save_context({"input": context}, {"output": output})


    # def save(self, context, output):
    #     self.retriever.save_context({"input": context}, {"output": output})
    #



