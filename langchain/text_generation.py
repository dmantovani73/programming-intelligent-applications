from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from client import get_client

# Get client
client = get_client("google")

# Prompt
prompt = "Spiega brevemente il concetto di machine learning in italiano."
messages = [HumanMessage(content=prompt)]

# Text completion
response = client.invoke(messages)
print(f"Text completion: {response.content}")

# Text completion (streaming)
stream = client.stream(messages)
print("\nStreaming:")
for chunk in stream:
    print(chunk.content, end="", flush=True)

# Chat prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Sei un esperto in machine learning, rispondi sempre in italiano, dai spiegazioni sintetiche e concise comprensibili anche a non esperti."),
    ("human", "{input}"),
])

# Chat
chat = prompt_template | client # chain espressa in LCEL (LangChain Express Language)
stream = chat.stream({"input": "Machine Learning"})
print("\n\nPromptTemplate:")
for chunk in stream:
    print(chunk.content, end="", flush=True)