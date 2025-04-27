import base64
from openai import OpenAI
import os
from dotenv import load_dotenv
import requests

load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

# Configure the client for Mistral AI
client = OpenAI(
    api_key=api_key,
    base_url="https://api.mistral.ai/v1",
)

prompt = "What is in this image?"
img_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d5/2023_06_08_Raccoon1.jpg/1599px-2023_06_08_Raccoon1.jpg"
    
response = client.chat.completions.create(
    model="mistral-small-latest",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": img_url},
            ],
        }
    ],
)

print(response.choices[0].message.content)