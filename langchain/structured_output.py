from typing import Optional
from client import get_client
from pydantic import BaseModel, Field

class Persona(BaseModel):
    """Informazioni su una persona."""
    nome: str = Field(description="Il nome della persona.")
    cognome: str = Field(description="Il cognome della persona.")
    eta: Optional[int] = Field(description="L'età della persona.")
    citta: Optional[str] = Field(description="La città di residenza.")

prompt = "Estrai le informazioni su Mario Rossi, che ha 30 anni e vive a Roma."

client = get_client("google")
client = client.with_structured_output(Persona)

response = client.invoke(prompt)
print(response)
