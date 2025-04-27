from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

MODEL = "mistral-small-latest"

# Configure the client for Mistral AI
client = OpenAI(
    api_key=api_key,
    base_url="https://api.mistral.ai/v1",
)

chat_history = []

def complete(prompt):
    chat_history.append({"role": "user", "content": prompt})
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=chat_history,
        stream=True,
    )

    full_content = ""
    
    print("Assistant: ", end="", flush=True)
    for chunk in response:
        content_delta = chunk.choices[0].delta.content
        if content_delta is not None:
            print(content_delta, end="", flush=True)
            full_content += content_delta
    
    print()  # New line after response is complete
    
    chat_history.append({"role": "assistant", "content": full_content})
    return full_content

while True:
    try:
        prompt = input("You: ")
        complete(prompt)
        print()
    except KeyboardInterrupt:
        break

