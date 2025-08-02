# chains/reasoning_chain.py
# from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
# Example: reuse your existing OpenAI setup
from google import genai

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
llm = genai.Client(api_key="AIzaSyCGmBYgzEah2oY0JPTPWVyeHwC-qk3NpsA")

def create_reasoning_chain(memory_retriever):
    prompt = PromptTemplate(
        input_variables=["context"],
        template=open("prompts/prompt_template.txt").read()
    )

    # llm = ChatOpenAI(temperature=0.7)
    chain = LLMChain(llm=llm, prompt=prompt, memory=memory_retriever)
    return chain




