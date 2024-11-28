# https://ollama.com/blog/embedding-models
# https://ollama.com/search?c=embedding

# ollama pull all-minilm

import ollama
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

# Frasi da calcolare
a = "La partita di scacchi era molto intensa."
b = "Il torneo di scacchi è stato avvincente."
c = "Il sole splendeva luminoso nel cielo."
d = "La competizione di scacchi è stata molto serrata."
e = "I bambini giocavano felicemente nel parco."
f = "L'incontro di calcio è stato emozionante."
g = "Il computer è acceso e sta eseguendo un programma."
h = "Il cielo era coperto di nuvole scure e minacciose."
i = "Il sistema operativo del computer si stava aggiornando."
j = "La scacchiera era pronta per l'inizio della partita."
k = "L'aria fresca di montagna riempiva i polmoni."
l = "Gli spettatori erano in attesa del primo colpo della partita."
m = "Il fiore era colorato e pieno di vita."
n = "La partita di tennis si è conclusa con un match point incredibile."

# Inizializza il client Ollama
client = ollama.Client()


def get_embedding(prompt, model="nomic-embed-text"):
    return client.embeddings(model=model, prompt=prompt).embedding


def get_cosine_distance(a, b):
    a_embedding = get_embedding(a)
    b_embedding = get_embedding(b)
    a_vector = np.array(a_embedding).reshape(1, -1)
    b_vector = np.array(b_embedding).reshape(1, -1)
    return cosine_distances(a_vector, b_vector)[0][0]


# Calcola la distanza del coseno tra diverse frasi utilizzando sklearn
distance_a_b = get_cosine_distance(a, b)
distance_a_c = get_cosine_distance(a, c)
distance_a_d = get_cosine_distance(a, d)
distance_a_e = get_cosine_distance(a, e)
distance_a_f = get_cosine_distance(a, f)
distance_g_i = get_cosine_distance(g, i)
distance_g_h = get_cosine_distance(g, h)
distance_j_l = get_cosine_distance(j, l)
distance_k_m = get_cosine_distance(k, m)

print(f"Distanza del coseno tra 'a' e 'b': {distance_a_b}")
print(f"Distanza del coseno tra 'a' e 'c': {distance_a_c}")
print(f"Distanza del coseno tra 'a' e 'd': {distance_a_d}")
print(f"Distanza del coseno tra 'a' e 'e': {distance_a_e}")
print(f"Distanza del coseno tra 'a' e 'f': {distance_a_f}")
print(f"Distanza del coseno tra 'g' e 'i': {distance_g_i}")
print(f"Distanza del coseno tra 'g' e 'h': {distance_g_h}")
print(f"Distanza del coseno tra 'j' e 'l': {distance_j_l}")
print(f"Distanza del coseno tra 'k' e 'm': {distance_k_m}")
