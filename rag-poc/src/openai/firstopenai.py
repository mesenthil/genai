from openai import  OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

#print(OPENAI_API_KEY)
client=OpenAI()
prompt="please suggest ways to learn GenAI"
completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
)

print(completion.choices[0].message)
