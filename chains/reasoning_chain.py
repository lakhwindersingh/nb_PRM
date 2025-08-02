# chains/reasoning_chain.py
# from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
# Example: reuse your existing OpenAI setup
from openai import OpenAI

# Point to the local server
llm = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

def create_reasoning_chain(memory_retriever):
    prompt = PromptTemplate(
        input_variables=["context"],
        template=open("prompts/prompt_template.txt").read()
    )

    # llm = ChatOpenAI(temperature=0.7)
    chain = LLMChain(llm=llm, prompt=prompt, memory=memory_retriever)
    return chain




