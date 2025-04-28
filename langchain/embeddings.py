from client import get_embeddings

text_to_embed = "L'intelligenza artificiale sta trasformando il mondo."
texts_to_embed = ["Questo è il primo documento.", "Questo è il secondo documento."]

# Get embeddings
embeddings = get_embeddings("ollama")

# Embed text
vector_single = embeddings.embed_query(text_to_embed)
vectors_batch = embeddings.embed_documents(texts_to_embed)

print(f"Vector single: {vector_single}")
print(f"Vectors batch: {vectors_batch}")