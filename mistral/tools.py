from openai import OpenAI
import os
from dotenv import load_dotenv
import requests
import json
import sys
import urllib.parse

load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

MODEL = "mistral-small-latest"

# Configure the client for Mistral AI
client = OpenAI(
    api_key=api_key,
    base_url="https://api.mistral.ai/v1",
)

# Dichiaro le funzioni che implementano i tools
def get_coordinates(city):
    # Properly encode the city name for the URL
    encoded_city = urllib.parse.quote(city)
    url = f"https://nominatim.openstreetmap.org/search?q={encoded_city}&format=json"
    
    # Add a user agent to avoid being blocked
    headers = {
        "User-Agent": "ProgrammingIntelligentApplications/1.0"
    }
    
    try:
        response = requests.get(url, headers=headers)
        # Check if the request was successful
        response.raise_for_status()
        
        # Check if there's content in the response
        if not response.text:
            print(f"[ERROR] Empty response from API for city: {city}")
            return "Error: No data received from geolocation service"
            
        # Parse the JSON response
        data = response.json()
        
        # Check if we got any results
        if not data:
            print(f"[ERROR] No location data found for city: {city}")
            return "Error: City not found"
            
        return data[0]['lat'], data[0]['lon']
        
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Request failed: {e}")
        return f"Error: Failed to get coordinates - {str(e)}"
    except (json.JSONDecodeError, KeyError) as e:
        print(f"[ERROR] Parsing failed: {e}")
        print(f"[ERROR] Response content: {response.text[:100]}...")
        return f"Error: Failed to parse location data - {str(e)}"

def get_weather(city):
    try:
        coordinates = get_coordinates(city)
        latitude, longitude = coordinates

        response = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m")
        response.raise_for_status()
        data = response.json()
        return data['current']['temperature_2m']
    except Exception as e:
        print(f"[ERROR] Weather API error: {e}")
        return f"Error: Failed to get weather data - {str(e)}"

# Dichiaro i tools che voglio usare
tools = [    
    {
        "function": {
            "name": "get_weather",
            "description": "Get current temperature for provided coordinates in celsius.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The city to get weather for."}
                },
                "required": ["city"]
            }
        }
    }
]

print("What's the weather like in: ", end="")
city = input()

prompt = f"What's the weather like in {city} today?"

chat_history = [{"role": "user", "content": prompt}]

# La prima richiesta sarà una di tipo tool calling
response = client.chat.completions.create(
    model=MODEL,   
    messages=chat_history,
    tools=tools,
)

# Estrai il messaggio dalla risposta
message = response.choices[0].message

# Aggiungi il messaggio dell'assistente alla chat history
chat_history.append({"role": "assistant", "content": None, "tool_calls": message.tool_calls})

if not message.tool_calls:
    print("*** Unexpected response from the model! ***")
    print(response)
    sys.exit(0)

# Invoca il tool
tool_call = message.tool_calls[0]
function_name = tool_call.function.name
function_args = json.loads(tool_call.function.arguments)
result = get_weather(**function_args)

# Aggiungi la risposta del tool alla chat history
chat_history.append({
    "role": "tool",
    "tool_call_id": tool_call.id,
    "content": str(result)
})

# Invoca il modello di nuovo per ottenere la risposta finale
response = client.chat.completions.create(
    model=MODEL,
    messages=chat_history,
)

# Estrai il messaggio dalla risposta
message = response.choices[0].message

print(message.content)