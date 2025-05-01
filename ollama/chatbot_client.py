from gradio_client import Client

client = Client("http://127.0.0.1:7860/")
result = client.predict(
    message="Hi!!", param_2="smollm2:latest", param_3=0.7, api_name="/chat"
)
print(result)
