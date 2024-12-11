from abc import ABC, abstractmethod
from fastapi import HTTPException
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from llama_index.llms.openai import OpenAI as OpenAILlama
from langchain_core.prompts.prompt import PromptTemplate
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import SimpleChatEngine

from config import OPENAI_API_KEY, MODEL, REDIS_URL
from prompts import SYSTEM_PROMPT

client_openai = OpenAI(api_key=OPENAI_API_KEY)
client_langchain = ChatOpenAI(model=MODEL, api_key=OPENAI_API_KEY)
client_llamaindex = OpenAILlama(model=MODEL, api_key=OPENAI_API_KEY)

import redis


class AnswerGenerator(ABC):
    def __init__(
        self,
    ):
        self.SYSTEM_PROMPT = SYSTEM_PROMPT
        self.message_history = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        self.session_id = ""
        self.redis_client = redis.StrictRedis(
            host="redis-stack", port=6379, decode_responses=True
        )
    
    def set_index(self, session_name,  session_id: str, index: str):
        try:
            existing_data = self.redis_client.get(index)
            if existing_data:
                all_sessions_data = eval(existing_data)
            else:
                all_sessions_data = {}
            all_sessions_data[session_name] = session_id
            self.redis_client.set(index, str(all_sessions_data))
        except Exception as e:
            print(str(e))
            raise HTTPException(
                status_code=500, detail=f"Redis error: {str(e)}"
            )

    def save_to_redis(self):
        """
        Save the message history to Redis under the session_id key.
        """
        try:
            self.redis_client.set(self.session_id, str(self.message_history))
        except Exception as e:
            print(str(e))
            raise HTTPException(status_code=500, detail=f"Redis error: {str(e)}")

    def load_from_redis(self):
        """
        Load the message history from Redis if it exists.
        If not, initialize a new session with default message history.
        """

        try:
            history = self.redis_client.get(self.session_id)

            if history:
                self.message_history = eval(history)
            else:
                self.redis_client.setnx(self.session_id, str(self.message_history))
        except Exception as e:
            print(str(e))
            raise HTTPException(status_code=500, detail=f"Redis error: {str(e)}")

    @abstractmethod
    def generate_answer(self, question: str):
        """
        Abstract method to generate an answer.
        """
        pass