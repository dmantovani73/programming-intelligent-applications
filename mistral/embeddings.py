from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.mistral.ai/v1",
)

# Compute embeddings using Mistral's embedding model
response = client.embeddings.create(
    model="mistral-embed",  # Mistral's embedding model
    input="Hello world",    # The text to embed
)

# Get the embedding vector
embedding = response.data[0].embedding

print(f"Embedding dimension: {len(embedding)}")
print(f"First few values: {embedding[:5]}")
