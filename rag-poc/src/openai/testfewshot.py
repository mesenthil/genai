#https://colab.research.google.com/drive/1BkpMLfYEofhNK-PCKCSj9_SJqnUK40gR?usp=sharing#scrollTo=ihj7fUsDxTGb
from langchain.llms import OpenAI
import os
from dotenv import load_dotenv
from langchain import PromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain

load_dotenv()
OPEN_API_KEY=os.getenv("OPENAI_API_KEY")
#llm = OpenAI(temperature=0.9)
llm = OpenAI(temperature=0.7)

example =[
    {"word":"happy", "antonym":"sad"},
    {"word":"tall", "antonym":"short"}
]
example_formater="""
Word:{word} \n
ANTONYM:{antonym}
"""

example_prompt_template=PromptTemplate(
    input_variables=["word","antonym"],
    template=example_formater
)

few_shot_prompt=FewShotPromptTemplate(
    examples=example,
    example_prompt=example_prompt_template,
    prefix="please print the antonym of every input ",
    suffix="Word: {ip} \n Antonym : ",
    input_variables=["ip"]

)

print(few_shot_prompt.format(ip="big"))
chain=LLMChain(llm=llm, prompt=few_shot_prompt)

print(chain.invoke("man is big "))