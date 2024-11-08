SYSTEM_PROMPT = (
    "You are a helpful assistant which answers general question asked by a user."
)
SYSTEM_PROMPT_LANGCHAIN = """
You are a helpful assistant which answers general question asked by a user.
Current conversation:
{history}
Human: {input}
AI Assistant:
"""
