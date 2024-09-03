from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

llm = OpenAI(temperature=0.7)

my_prompt_template=PromptTemplate(
    input_variable=["restaurent_description"],
    template="""
    I want you to act as a naming consultant for few restaurents.
    Return the list of restaurent names. each should be short and easy to remember.
    
    what are the names for a restaurent that is {senthil}
    """
)

description="authentic indian restaurent that serves idli and sambar"
#my_prompt_template.format(rest_description=description)
chain=LLMChain(llm=llm,prompt=my_prompt_template)
print(chain.run(description))