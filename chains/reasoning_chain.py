# chains/reasoning_chain.py
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

def create_reasoning_chain(memory_retriever):
    prompt = PromptTemplate(
        input_variables=["context"],
        template=open("prompts/prompt_template.txt").read()
    )

    llm = ChatOpenAI(temperature=0.7)
    chain = LLMChain(llm=llm, prompt=prompt, memory=memory_retriever)
    return chain
