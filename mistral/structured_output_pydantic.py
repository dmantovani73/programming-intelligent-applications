from openai import OpenAI
import os
from dotenv import load_dotenv
from pydantic import BaseModel
import json

load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.mistral.ai/v1",
)

# Define a Pydantic model for the movie data
class Movie(BaseModel):
    title: str
    director: str
    year: int
    plot: str

# Example of getting structured JSON output from Mistral
response = client.chat.completions.create(
    model="mistral-small",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant that provides information about movies in a valid JSON format. Return only valid JSON with no extra text."
        },
        {
            "role": "user",
            "content": "Give me information about the movie Inception with this exact JSON structure: {\"title\": \"movie title\", \"director\": \"director name\", \"year\": year as integer, \"plot\": \"brief plot summary\"}"
        }
    ],
    response_format={"type": "json_object"},  # Usa response_format invece di text_format
)

# Get the JSON response as text
json_response = response.choices[0].message.content

# Parse the JSON response
movie_data = json.loads(json_response)

# Validate with Pydantic
movie = Movie(**movie_data)

# Print the validated object
print(movie)