from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain

repo_id = "google/flan-t5-xl" 

llm = HuggingFaceHub(
    huggingfacehub_api_token="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", 
    repo_id=repo_id, 
    model_kwargs={"temperature": 0.01, "max_length": 64}
)

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "Who won the FIFA World Cup in the year 1994?"

print(llm_chain.run(question))