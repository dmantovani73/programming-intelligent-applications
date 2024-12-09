# https://ollama.com/blog/structured-outputs
# pip install -U ollama

from typing import List, Literal, Optional
from ollama import chat
from pydantic import BaseModel
from openai import OpenAI
import openai


def structured_output():
    """
    Fetches and prints structured information about a country using a chat model.
    This function defines a Country model with attributes for the country's name,
    capital, and languages. It then sends a request to a chat model to get information
    about Canada, validates the response against the Country model, and prints the
    resulting Country object.
    The chat model used is "smollm2" and the response is expected to be in JSON format
    that adheres to the schema defined by the Country model.
    Returns:
        None
    """

    class Country(BaseModel):
        name: str
        capital: str
        languages: list[str]

    response = chat(
        messages=[
            {
                "role": "user",
                "content": "Tell me about Canada.",
            }
        ],
        model="smollm2",
        format=Country.model_json_schema(),
    )

    country = Country.model_validate_json(response.message.content)

    print("Structured output:")
    print(country)


def data_extraction():
    """
    Extracts pet data from a chat response and prints the extracted data.
    This function defines two classes, `Pet` and `PetList`, using Pydantic's `BaseModel`.
    It then sends a chat message to a model to extract information about pets from a given text.
    The response is validated and parsed into a `PetList` object, which is then printed.
    Classes:
        Pet: A Pydantic model representing a pet with attributes:
            - name (str): The name of the pet.
            - animal (str): The type of animal.
            - age (int): The age of the pet.
            - color (str | None): The color of the pet (optional).
            - favorite_toy (str | None): The pet's favorite toy (optional).
        PetList: A Pydantic model representing a list of pets.
    The function performs the following steps:
        1. Defines the `Pet` and `PetList` classes.
        2. Sends a chat message to the model with a predefined prompt.
        3. Validates and parses the response into a `PetList` object.
        4. Prints the extracted pet data.
    Note:
        The `chat` function and `smollm2` model are assumed to be defined elsewhere in the codebase.
    """

    class Pet(BaseModel):
        name: str
        animal: str
        age: int
        color: str | None
        favorite_toy: str | None

    class PetList(BaseModel):
        pets: list[Pet]

    response = chat(
        messages=[
            {
                "role": "user",
                "content": """
                    I have two pets.
                    A cat named Luna who is 5 years old and loves playing with yarn. She has grey fur.
                    I also have a 2 year old black cat named Loki who loves tennis balls.
                """,
            }
        ],
        model="smollm2",
        format=PetList.model_json_schema(),
    )

    pets = PetList.model_validate_json(response.message.content)

    print("Data extraction:")
    print(pets)


def image_description():
    """
    Analyzes an image and provides a detailed description including objects, scene, colors,
    time of day, setting, and any detected text content.
    The function uses a pre-trained vision model to analyze the image and returns a structured
    description based on the provided schema.
    Classes:
        Object: Represents an object detected in the image with its name, confidence score,
                and attributes.
        ImageDescription: Represents the overall description of the image including a summary,
                          list of detected objects, scene, colors, time of day, setting, and
                          optional text content.
    The function performs the following steps:
    1. Defines the Object and ImageDescription classes based on the provided schema.
    2. Sets the path to the image to be analyzed.
    3. Sends a request to the vision model with the image and the schema for the response.
    4. Validates the response against the ImageDescription schema.
    5. Prints the structured image description.
    Returns:
        None
    """

    class Object(BaseModel):
        name: str
        confidence: float
        attributes: str

    class ImageDescription(BaseModel):
        summary: str
        objects: List[Object]
        scene: str
        colors: List[str]
        time_of_day: Literal["Morning", "Afternoon", "Evening", "Night"]
        setting: Literal["Indoor", "Outdoor", "Unknown"]
        text_content: Optional[str] = None

    path = "beach.jpg"

    response = chat(
        model="llama3.2-vision",
        format=ImageDescription.model_json_schema(),  # Pass in the schema for the response
        messages=[
            {
                "role": "user",
                "content": "Analyze this image and describe what you see, including any objects, the scene, colors and any text you can detect.",
                "images": [path],
            },
        ],
        options={
            "temperature": 0
        },  # Set temperature to 0 for more deterministic output
    )

    image_description = ImageDescription.model_validate_json(response.message.content)

    print("Image description:")
    print(image_description)


def openai_compatibility():
    """
    Demonstrates compatibility with OpenAI's API by creating a client and sending a chat completion request.
    The function defines two Pydantic models, `Pet` and `PetList`, to structure the response data. It then sends a
    chat completion request to the OpenAI API with a predefined message about two pets. The response is expected
    to be parsed into the `PetList` model.
    If the response is successfully parsed, it prints the parsed data. If the response contains a refusal, it prints
    the refusal message. In case of an exception, it handles specific `openai.LengthFinishReasonError` for token
    length issues and prints the error message for other exceptions.
    Raises:
        openai.LengthFinishReasonError: If the response contains too many tokens.
        Exception: For other exceptions that may occur during the API request.
    """
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    class Pet(BaseModel):
        name: str
        animal: str
        age: int
        color: str | None
        favorite_toy: str | None

    class PetList(BaseModel):
        pets: list[Pet]

    try:
        completion = client.beta.chat.completions.parse(
            temperature=0,
            model="smollm2",
            messages=[
                {
                    "role": "user",
                    "content": """
                        I have two pets.
                        A cat named Luna who is 5 years old and loves playing with yarn. She has grey fur.
                        I also have a 2 year old black cat named Loki who loves tennis balls.
                    """,
                }
            ],
            response_format=PetList,
        )

        print("OpenAI compatibility:")
        pet_response = completion.choices[0].message
        if pet_response.parsed:
            print(pet_response.parsed)
        elif pet_response.refusal:
            print(pet_response.refusal)
    except Exception as e:
        if type(e) == openai.LengthFinishReasonError:
            print("Too many tokens: ", e)
            pass
        else:
            print(e)
            pass


structured_output()

print()
data_extraction()

print()
openai_compatibility()

print()
image_description()
