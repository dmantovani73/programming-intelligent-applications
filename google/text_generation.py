# https://ai.google.dev/gemini-api/docs/

from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
MODEL = "gemini-2.0-flash"

client = genai.Client(api_key=api_key)

prompt = "How does AI work?"

response = client.models.generate_content(
    model=MODEL, contents=[prompt]
)

print(response.text)


# Streaming
response = client.models.generate_content_stream(
    model=MODEL,
    contents=prompt
)
for chunk in response:
    print(chunk.text, end="")