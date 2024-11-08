from abc import ABC, abstractmethod
from fastapi import HTTPException
from fastapi import HTTPException
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI as OpenAILlama
from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain
from langchain_core.prompts.prompt import PromptTemplate
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import SimpleChatEngine

from config import OPENAI_API_KEY, MODEL
from prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_LANGCHAIN
from llama_index.core.memory import ChatSummaryMemoryBuffer

client_openai = OpenAI(api_key=OPENAI_API_KEY)
client_langchain = ChatOpenAI(model=MODEL, api_key=OPENAI_API_KEY)
client_llamaindex = OpenAILlama(model=MODEL, api_key=OPENAI_API_KEY)

SYSTEM_PROMPT_LANGCHAIN = PromptTemplate(
    input_variables=["history", "input"], template=SYSTEM_PROMPT_LANGCHAIN
)
conversation = ConversationChain(
    prompt=SYSTEM_PROMPT_LANGCHAIN,
    llm=client_langchain,
    memory=ConversationSummaryMemory(llm=client_langchain),
)

memory = ChatSummaryMemoryBuffer.from_defaults()
chat_engine = SimpleChatEngine.from_defaults(memory=memory, llm=client_llamaindex)


class AnswerGenerator(ABC):
    def __init__(self):
        self.SYSTEM_PROMPT = SYSTEM_PROMPT
        self.message_history = [{"role": "system", "content": self.SYSTEM_PROMPT}]

    @abstractmethod
    def generate_answer(self, question: str):
        """
        Abstract method to generate an answer.
        """
        pass


class OpenAIAnswerGenerator(AnswerGenerator):
    def generate_answer(self, question: str):
        try:
            self.message_history.append({"role": "user", "content": question})
            answer = client_openai.chat.completions.create(
                model=MODEL, messages=self.message_history
            )
            final_answer = answer.choices[0].message.content
            self.message_history.append({"role": "assistant", "content": final_answer})
            return final_answer
        except Exception as e:
            raise HTTPException(
                status_code=500, detail="Error generating answer with OpenAI"
            )


class LangChainAnswerGenerator(AnswerGenerator):
    def generate_answer(self, question: str):
        try:
            answer = conversation.invoke(question)
            return answer["response"]

        except Exception as e:
            raise HTTPException(
                status_code=500, detail="Error generating answer with LangChain"
            )


class LlamaIndexAnswerGenerator(AnswerGenerator):
    def generate_answer(self, question: str):
        try:
            answer = chat_engine.chat(question)
            print(answer)
            return str(answer)
        except Exception as e:
            print(str(e))
            raise HTTPException(
                status_code=500, detail="Error generating answer with Llama Index"
            )
