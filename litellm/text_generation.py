# https://docs.litellm.ai/docs/

from litellm import completion
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

MODEL = "openrouter/google/gemma-3-12b-it:free"

# Make a call to the model
response = completion(
  model=MODEL,
  messages = [{ "content": "Spiegami il prompt engineering in breve.","role": "user"}],
)

print(response)