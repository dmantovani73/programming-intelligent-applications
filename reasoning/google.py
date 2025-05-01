# https://ai.google.dev/gemini-api/docs/openai?hl=it#thinking

from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

client = OpenAI(
    api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# low = 1000 tokens
# medium = 8000 tokens
# high = 24000 tokens
reasoning_effort = "low"

response = client.chat.completions.create(
    model="gemini-2.5-flash-preview-04-17",
    reasoning_effort=reasoning_effort,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain to me how AI works"},
    ],
    stream=True,
)

for chunk in response:
    print(chunk.choices[0].delta.content)
