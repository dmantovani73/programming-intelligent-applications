from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings

load_dotenv()

def get_client(provider: str, model: str = None) -> BaseChatModel:
    """
    Creates and returns a chat model client for the specified provider.
    
    Args:
        provider: The LLM provider name ('google', 'mistral', or 'ollama')
        model: Optional model name. If not provided, uses a default model for the provider.
    
    Returns:
        BaseChatModel: A configured chat model client for the specified provider.
    
    Raises:
        ValueError: If an unsupported provider is specified.
    """
    if provider == "google":
        return ChatGoogleGenerativeAI(model=model or "gemini-1.5-flash-latest")
    elif provider == "mistral":
        return ChatMistralAI(model=model or "mistral-small-latest")
    elif provider == "ollama":
        return ChatOllama(model=model or "phi4-mini:latest")
    
def get_embeddings(provider: str, model: str = None) -> Embeddings:
    """
    Creates and returns an embeddings client for the specified provider.
    
    Args:
        provider: The embeddings provider name ('google', 'mistral', or 'ollama')
        model: Optional model name. If not provided, uses a default model for the provider.
    
    Returns:
        BaseEmbedding: A configured embeddings client for the specified provider.
    
    Raises:
        ValueError: If an unsupported provider is specified.
    """
    if provider == "google":
        return GoogleGenerativeAIEmbeddings(model=model or "models/embedding-001")
    elif provider == "mistral":
        return MistralAIEmbeddings(model=model or "mistral-embed")
    elif provider == "ollama":
        return OllamaEmbeddings(model=model or "nomic-embed-text")
