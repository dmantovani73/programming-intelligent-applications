# https://ollama.com/search?c=tools

import ollama
import requests

model = "smollm2"


def add_two_numbers(a: int, b: int) -> int:
    """
    Add two numbers

    Args:
      a: The first integer number
      b: The second integer number

    Returns:
      int: The sum of the two numbers
    """
    return a + b


prompt = input("Enter your message: ")

response = ollama.chat(
    model,
    messages=[{"role": "user", "content": prompt}],
    tools=[add_two_numbers, requests.request],  # Actual function reference
)


available_functions = {
    "add_two_numbers": add_two_numbers,
    "request": requests.request,
}

for tool in response.message.tool_calls or []:
    function_to_call = available_functions.get(tool.function.name)

    if tool.function.name == "request":
        resp = function_to_call(
            method=tool.function.arguments.get("method"),
            url=tool.function.arguments.get("url"),
        )
        print(resp.text)
    else:
        if function_to_call:
            print("Function output:", function_to_call(**tool.function.arguments))
        else:
            print("Function not found:", tool.function.name)
