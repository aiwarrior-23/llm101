from abc import ABC, abstractmethod
from fastapi import HTTPException
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from llama_index.llms.openai import OpenAI as OpenAILlama
from langchain_core.prompts.prompt import PromptTemplate
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import SimpleChatEngine
from langchain.graphs import Neo4jGraph
from neo4j import GraphDatabase
import json

from config import *
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
            host="localhost", port=6379, decode_responses=True
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

class DataAnswerGenerator(ABC):
    def __init__(
        self,
    ):
        self.data_path = "" # here we will store the path where csv is uploaded
        self.data = "" # here we will store the complete data
        self.database = "" # here we will store the name of knowledge graph db
        self.graph = "" # here we will store the reference to knowledge graph
        self.insert_query = "" # here we will store the query used to insert relationships in graph db
        self.cache = {"message_history":[]} # here we will store the conversation history
        self.session_id = "" # store the new session id or reuse older session
        self.redis_client = redis.StrictRedis(
            host="localhost", port=6379, decode_responses=True
        ) # connect with redis db for session management

    def sanitize(self, text):
        """
            In this function we will process the LLM response.
        """
        return str(text).replace("'", "").replace('"', "").replace("{", "").replace("}", "")

    def connect_to_db(self):
        """
            This function is used to connect with graph db
        """
        url = NEO4J_URL
        username = NEO4J_UNAME
        password = NEO4J_PASS
        graph = Neo4jGraph(
            url=url, 
            username=username, 
            password=password, 
            database=self.database
        )
        self.graph = graph

    def create_database(self):
        """
            This function will be used to create a new db or use old db
        """
        
        # Connect to neo4j
        driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_UNAME, NEO4J_PASS))

        try:
            with driver.session() as session:
                # check if db exists
                result = session.run(f"SHOW DATABASES WHERE name='{self.database}'")
                # if exists return nothing
                if result.single():
                    return
                
                # if not exists, create the database
                session.run(f"CREATE DATABASE {self.database}")
                return "success"
        except Exception as e:
            print(str(e))
            return "failure"
        finally:
            driver.close()

    def get_prompt(self, template, **kwargs):
        input_variables = kwargs["input_variables"]
        replacements = kwargs["replacements"]
        if len(replacements) > 0:
            for k,v in replacements.items():
                template = template.replace(k, v.replace('{', '{{').replace('}', '}}'))
        
        if len(input_variables) == 0:
            prompt = template
        
        else:
            prompt = PromptTemplate(
                template=template,
                input_variables=input_variables
            )
        return prompt
    
    def set_index(self, session_name,  session_id: str, index: str):
        try:
            existing_data = self.redis_client.get(index)
            if existing_data:
                all_sessions_data = eval(existing_data)
            else:
                all_sessions_data = {}
            all_sessions_data[session_name] = session_id
            print("storing to redis")
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
            self.redis_client.set(self.session_id, json.dumps(self.cache))
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
                self.cache = eval(history)
            else:
                self.redis_client.setnx(self.session_id, json.dumps(self.cache))
        except Exception as e:
            print(str(e))
            raise HTTPException(status_code=500, detail=f"Redis error: {str(e)}")
    
    @abstractmethod
    def load_data(self):
        pass


    @abstractmethod
    def get_results(self):
        pass

    @abstractmethod
    def dump_to_cache(self):
        pass

    @abstractmethod
    def generate_answer_from_data(self):
        pass