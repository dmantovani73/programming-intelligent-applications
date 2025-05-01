import ollama

available_models = [m["model"] for m in ollama.list()["models"]]
print(available_models)
