from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1",
)

response = client.chat.completions.create(
    model="llama3-8b-8192", 
    messages=[
        {"role": "system", "content": "Sei un assistente utile."},
        {"role": "user", "content": "Spiegami il prompt engineering in breve."},
    ],
)

print(response)