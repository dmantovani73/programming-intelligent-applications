import base64
import requests
from client import get_client
from langchain_core.messages import HumanMessage

# client = get_client("google")
# client = get_client("mistral")
client = get_client("ollama", model="gemma3:27b")


def load_image_from_url(image_url: str) -> bytes | None:
    """Scarica un'immagine da un URL e la restituisce come bytes."""
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()  # Solleva un errore per status code non validi (es. 404)
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"Errore durante il download dell'immagine da {image_url}: {e}")
        return None


def encode_image(image_bytes: bytes) -> str:
    """Codifica i bytes di un'immagine in base64."""
    return base64.b64encode(image_bytes).decode("utf-8")


def ask_image_question(image_url: str, question: str):
    """Fai una domanda su un'immagine data un URL e una domanda."""
    image_bytes = load_image_from_url(image_url)
    if image_bytes is None:
        print("Impossibile caricare l'immagine.")
        return

    base64_image = encode_image(image_bytes)
    image_data_uri = f"data:image/jpeg;base64,{base64_image}"  # Assumiamo JPEG, potrebbe essere necessario adattarlo

    message = HumanMessage(
        content=[
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": image_data_uri}},
        ]
    )

    response = client.invoke([message])
    print(f"Risposta: {response.content}")


prompt = "Descrivi cosa vedi in questa immagine"
ask_image_question(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg",
    prompt,
)

print()

ask_image_question(
    "https://images.pexels.com/photos/374870/pexels-photo-374870.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1",
    prompt,
)
