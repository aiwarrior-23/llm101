from abc import ABC, abstractmethod

from fastapi import HTTPException

from openai import OpenAI
from langchain_openai import ChatOpenAI

from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import RedisChatMessageHistory

from llama_index.core.memory import ChatMemoryBuffer, ChatSummaryMemoryBuffer
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.storage.chat_store.redis import RedisChatStore
from llama_index.llms.openai import OpenAI as OpenAILlama

from config import OPENAI_API_KEY, MODEL, REDIS_URL
from prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_LANGCHAIN, ner_prompt_generator
from answer_generator import AnswerGenerator

import redis

client_openai = OpenAI(api_key=OPENAI_API_KEY)
client_langchain = ChatOpenAI(model=MODEL, api_key=OPENAI_API_KEY)
client_llamaindex = OpenAILlama(model=MODEL, api_key=OPENAI_API_KEY)


class OpenAIAnswerGenerator(AnswerGenerator):
    def generate_answer(self, question: str, session_id: str):
        try:
            if self.session_id != session_id:
                self.session_id = session_id
                try:
                    session_name = client_openai.chat.completions.create(model=MODEL, messages=ner_prompt_generator(question, "openai"))
                    self.set_index(session_name.choices[0].message.content,session_id,"all_sessions_data_openai")
                except Exception as e:
                    print(str(e))
                    
            self.load_from_redis()
            self.message_history.append({"role": "user", "content": question})
            
            answer = client_openai.chat.completions.create(model=MODEL, messages=self.message_history)
            
            self.message_history.append({"role": "assistant", "content": answer.choices[0].message.content})
            self.save_to_redis()

            return answer.choices[0].message.content
        
        except Exception as e:
            print(str(e))
            raise HTTPException(
                status_code=500, detail="Error generating answer with OpenAI"
            )


class LangChainAnswerGenerator(AnswerGenerator):
    def __init__(self):
        super().__init__()
        self.chain = SYSTEM_PROMPT_LANGCHAIN | client_langchain
    
    def get_redis_history(self, session_id: str):
        return RedisChatMessageHistory(session_id, url=REDIS_URL)

    def generate_answer(self, question: str, session_id: str):
        try:
            print(question, session_id, self.session_id)
            if self.session_id != session_id:
                self.session_id = session_id
                session_name = client_langchain.invoke(ner_prompt_generator(question, "langchain"))
                self.set_index(session_name.content, session_id, "all_sessions_data_langchain")
            conversation = RunnableWithMessageHistory(self.chain, self.get_redis_history, input_messages_key="input", history_messages_key="history")
            answer = conversation.invoke({"input": question}, config={"configurable": {"session_id": session_id}})
            return answer.content

        except Exception as e:
            print(str(e))
            raise HTTPException(status_code=500, detail="Error generating answer with LangChain")


class LlamaIndexAnswerGenerator(AnswerGenerator):
    def __init__(self):
        super().__init__()
    
    def generate_answer(self, question: str, session_id: str):
        try:
            if self.session_id != session_id:
                self.session_id = session_id
                session_name = client_llamaindex.chat(ner_prompt_generator(question, "llamaindex"))
                self.set_index(str(session_name).replace("assistant: ", ""),session_id,"all_sessions_data_llama")
            
            chat_store = RedisChatStore(redis_url=REDIS_URL)
            chat_memory = ChatMemoryBuffer.from_defaults(chat_store=chat_store, chat_store_key=session_id)
            conversation = SimpleChatEngine.from_defaults(memory=chat_memory, llm=client_llamaindex)
            answer = conversation.chat(question)
            return answer.response
        
        except Exception as e:
            print(str(e))
            raise HTTPException(
                status_code=500, detail="Error generating answer with Llama Index"
            )
