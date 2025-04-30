from client import get_client


def complete(temperature: float):
    client = get_client("mistral", temperature=temperature)

    prompt = """Inventa una barzelletta completamente nuova e originale in italiano.
    La barzelletta deve essere diversa da quelle comuni come quelle sui pesci o sugli scheletri.
    Sii creativo e sorprendente."""

    stream = client.stream(prompt)

    print(f"\n\nRisposta (temperatura {temperature}):")
    for chunk in stream:
        print(chunk.content, end="", flush=True)


complete(0)
complete(0.5)
complete(1)
