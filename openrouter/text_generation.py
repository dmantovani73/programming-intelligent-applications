from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
)

response = client.chat.completions.create(
    model="google/gemma-3-12b-it:free",
    messages=[
        {"role": "system", "content": "Sei un assistente utile."},
        {"role": "user", "content": "Spiegami il prompt engineering in breve."},
    ],
)

print(response.choices[0].message.content)