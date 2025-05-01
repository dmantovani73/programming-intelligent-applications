import gradio as gr
import ollama
import time

try:
    # Add retry mechanism for connecting to Ollama
    max_retries = 3
    retry_count = 0
    while retry_count < max_retries:
        try:
            available_models = [m["model"] for m in ollama.list()["models"]]
            break
        except KeyError:
            # Handle case where response doesn't contain expected structure
            print(
                f"Warning: Unexpected response format from Ollama on attempt {retry_count+1}"
            )
            retry_count += 1
            time.sleep(1)
        except Exception as e:
            print(f"Error connecting to Ollama on attempt {retry_count+1}: {e}")
            retry_count += 1
            time.sleep(1)

    if retry_count >= max_retries:
        print("Failed to connect to Ollama after multiple attempts")
        available_models = ["Error loading models"]

    if not available_models:
        print("Error: No Ollama models found. Pull one using 'ollama pull model_name'.")
        available_models = ["No models available"]
        default_model = available_models[0]
    else:
        default_model = available_models[0]
        preferred_model = "llama3:latest"  # Or any other preferred model
        if preferred_model in available_models:
            default_model = preferred_model

except Exception as e:
    print(f"Error connecting to Ollama or fetching models: {e}")
    print("Ensure Ollama is running.")
    available_models = ["Error loading models"]
    default_model = available_models[0]


def ollama_chat(message, history, model_name, temperature):
    if model_name in ["No models available", "Error loading models"]:
        return f"Error: Please select a valid Ollama model. Found: {available_models}"

    messages = []
    for human_msg, ai_msg in history:
        if human_msg:
            messages.append({"role": "user", "content": human_msg})
        if ai_msg:
            messages.append({"role": "assistant", "content": ai_msg})

    messages.append({"role": "user", "content": message})

    try:
        response = ollama.chat(
            model=model_name,
            messages=messages,
            stream=False,
            options={"temperature": temperature},
        )

        content = response["message"]["content"]
        print(f"Response from {model_name}: {content}")

        return content
    except Exception as e:
        return f"An error occurred while communicating with Ollama: {e}"


with gr.Blocks(theme=gr.themes.Soft(), title="Ollama Chat") as demo:
    gr.Markdown("# Chat with Local Ollama Models")
    gr.Markdown(
        "Select an installed model and desired temperature, then start chatting."
    )

    with gr.Row():
        model_dropdown = gr.Dropdown(
            label="Select Ollama Model",
            choices=available_models,
            value=default_model,
            interactive=True,
        )
        temperature_slider = gr.Slider(
            label="Temperature",
            minimum=0.0,
            maximum=2.0,
            step=0.1,
            value=0.7,
            interactive=True,
        )

    chatbot = gr.ChatInterface(
        fn=ollama_chat,
        additional_inputs=[model_dropdown, temperature_slider],
    )

if __name__ == "__main__":
    demo.launch()
