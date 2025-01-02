from fastapi import FastAPI, Form
from pydantic import BaseModel
from typing import Dict
import pandas as pd
import json

# Import your existing functions
from test import *
from fastapi import FastAPI
from pydantic import BaseModel
from test import *
from llama_index.llms.openai import OpenAI as OpenAILlama
from fastapi.middleware.cors import CORSMiddleware
from app_session4 import OpenAIAnswerGenerator, LangChainAnswerGenerator, LlamaIndexAnswerGenerator, KGDataAnswerGenerator, PandasDataAnswerGenerator
import uuid
import redis
import json
from langchain_community.chat_message_histories import RedisChatMessageHistory
from llama_index.storage.chat_store.redis import RedisChatStore
from llama_index.core.memory import ChatMemoryBuffer
import re
from response_structure import SessionRequest, ChatHistoryRequest, QuestionRequest, QueryRequest
from config import *
from fastapi import UploadFile, File, HTTPException
import os

app = FastAPI()
kg_csv_data_generator = KGDataAnswerGenerator()
pandas_csv_data_generator = PandasDataAnswerGenerator()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model_openai = OpenAIAnswerGenerator()
model_langchain = LangChainAnswerGenerator()
model_llama = LlamaIndexAnswerGenerator()

redis_client = redis.StrictRedis(host='localhost', port=6379, decode_responses=True)
    
@app.get("/new_chat")
async def new_chat():
    """
        Endpoint to start a new chat session and generate a session_id
    """
    session_id = str(uuid.uuid4())
    return {"session_id": session_id}

@app.post("/get_sessions")
async def get_sessions(request: SessionRequest):
    model_type = request.model_type
    
    value = {}
    if model_type == "openai":
        value = redis_client.get("all_sessions_data_openai")
    elif model_type == "langchain":
        value = redis_client.get("all_sessions_data_langchain")
    elif model_type == "llamaindex":
        value = redis_client.get("all_sessions_data_llama")
    else:
        return {"error": "Invalid model type"}
    
    if value is None:
        return {"sessions_list": []}
    
    value = json.loads(value.replace("'", '"'))
    return {"sessions_list": list(value.keys())}

@app.post("/get_chat_history")
async def get_sessions(request: ChatHistoryRequest):
    session_name = request.session_name
    model_type = request.model_type

    if model_type == "openai":
        value = redis_client.get("all_sessions_data_openai")
        value = eval(value)
        session_id = value[session_name]
        value = redis_client.get(session_id)
        return {"chat_history": eval(value), "session_id": session_id}
    
    elif model_type == "langchain":
        value = redis_client.get("all_sessions_data_langchain")
        value = json.loads(value.replace("'", '"'))
        session_id = value[session_name]
        history = RedisChatMessageHistory(session_id=session_id, url="redis://localhost:6379", ttl=3600)
        value = await history.aget_messages()
        
        return {"chat_history": value, "session_id": session_id}
    
    elif model_type == "llamaindex":
        value = redis_client.get("all_sessions_data_llama")
        value = json.loads(value.replace("'", '"'))
        session_id = value[session_name]
        chat_store = RedisChatStore(redis_url="redis://localhost:6379")
        chat_memory = ChatMemoryBuffer.from_defaults(chat_store=chat_store,chat_store_key=session_id,)
        value = chat_memory.get()
        
        return {"chat_history": value, "session_id": session_id}
    
    elif model_type == "chat_with_csv":
        value = redis_client.get("all_sessions_data_langchain")
        value = eval(value)
        session_id = value[session_name]
        value = redis_client.get(session_id)
        return {"chat_history": eval(value), "session_id": session_id}
    
    else:
        return {"error": "Invalid model type"}

@app.post("/generate_answer")
async def generate_answer(question: str = Form(...),
    model_type: str = Form(...),
    session_id: str = Form(...),
    database_name: str = Form(...),
    regenerate: bool = Form(True),
    chat_type: str = Form(...),
    file: UploadFile | None = File(None)):
    
    data_path = None

    # check if any csv/excel sheet is attached with the request or not
    if file and (file.filename.lower().endswith(".csv") or file.filename.lower().endswith(".xlsx")):
        # if sheet is attached then a true boolean value
        chat_with_data = True
        # marks where the csv should be uploaded
        upload_dir = "/home/himanshu-singh/"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        data_path = file_path
    else:
        # if not present then false boolean value to run the previous flow
        chat_with_data = False

    if not chat_with_data:
        # this is the previous flow already explained.
        print("disabled chat with data mode")
        question = question
        session_id = session_id
        model_type = model_type.lower()

        if model_type == "openai":
            model = model_openai
        elif model_type == "langchain":
            model = model_langchain
        elif model_type == "llamaindex":
            model = model_llama
        else:
            raise HTTPException(status_code=400, detail="Invalid model_type. Choose from 'openai', 'langchain', or 'llamaindex'.")

        answer = model.generate_answer(question, session_id)
        return {"answer": answer}
    
    else:
        # get the boolean value. If true then a new graph db is created else existing is used
        regenerate = regenerate
        # which graph db to use/create
        database_name = database_name
        
        # if chat type is kg then initiate Knowledge Graph Chain else Pandas chain
        if chat_type == "kg":
            # send the details to this function so that the db can be created and question can be asked.
            print(model_type)
            answer = kg_csv_data_generator.generate_answer_from_data(query=question, session_id=session_id, regenerate=regenerate, data_path=data_path, database_name=database_name)
            return {"answer": answer}
        else:
            answer = pandas_csv_data_generator.generate_answer_from_data(query=question, session_id=session_id, regenerate=regenerate, data_path=data_path, database_name=database_name)
            return {"answer": answer}
