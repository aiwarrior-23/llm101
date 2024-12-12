from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from llama_index.core.llms import ChatMessage
from langchain_core.messages import HumanMessage, SystemMessage

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

SYSTEM_PROMPT_LANGCHAIN_2 = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)


def ner_prompt_generator(question, model_type):
    if model_type == "langchain":
        return [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=f"NER for the question in one or two     words. Give only the word and don't start it with some pre words like Named Entity - question - {question}"
            ),
        ]
    elif model_type == "llamaindex":
        return [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(
                role="user",
                content=f"NER for the question in one or two words. Give only the word and don't start it with some pre words like Named Entity - question - {question}",
            ),
        ]
    elif model_type == "openai":
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"NER for the question in one or two words. Give only the word and don't start it with some pre words like Named Entity - question - {question}",
            },
        ]
    else:
        raise ValueError("Invalid model type")
