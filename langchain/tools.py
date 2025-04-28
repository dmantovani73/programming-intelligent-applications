from client import get_client
import random as random_module  # Rename to avoid conflict with our tool function
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

tavily_tool = TavilySearchResults(max_results=1)

@tool
def random(min: int, max: int) -> int:
    """Random number between min and max."""
    return random_module.randint(min, max)

tools = [tavily_tool, random]  # Lista degli strumenti disponibili per il modello

client = get_client("mistral")
client = client.bind_tools(tools)

def complete(prompt: str):
    response = client.invoke(prompt)

    if response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_input = tool_call["args"]
            print(f"\nTool: {tool_name}, Input: {tool_input}")
            
            # Find the matching tool by name
            tool_to_invoke = next((t for t in tools if getattr(t, "name", None) == tool_name), None)
            
            if tool_to_invoke:
                # Execute the tool with the provided arguments
                try:
                    # Use the invoke method instead of calling the tool directly
                    result = tool_to_invoke.invoke(tool_input)
                    print(f"Tool result: {result}")
                except Exception as e:
                    print(f"Error invoking tool: {e}")
            else:
                print(f"Tool '{tool_name}' not found in available tools")
    else:
        print(f"\nResponse: {response.content}")

complete("Ciao!")
complete("Qual è il prezzo attuale delle azioni NVIDIA?")
complete("Genera un numero a caso tra 0 e 10")