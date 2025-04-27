from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.mistral.ai/v1",
)

# Example of getting structured JSON output from Mistral
response = client.chat.completions.create(
    model="mistral-small",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant that provides information about movies in JSON format."
        },
        {
            "role": "user", 
            "content": "Give me information about the movie Inception in JSON format with title, director, year, and brief plot."
        }
    ],
    response_format={"type": "json_object"}
)

# Get the JSON response
json_response = response.choices[0].message.content
print("Structured JSON output:")
print(json_response)

