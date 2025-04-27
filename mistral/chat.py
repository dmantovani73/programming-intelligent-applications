from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

# Configure the client for Mistral AI
client = OpenAI(
    api_key=api_key,
    base_url="https://api.mistral.ai/v1",
)

# Make a call to the model
response = client.chat.completions.create(
    model="mistral-small-latest",
    messages=[
        {"role": "system", "content": "Sei un assistente utile."},
        {"role": "user", "content": "Spiegami il prompt engineering in breve."},
    ],
)

print(response)