from fastapi import FastAPI
from pydantic import BaseModel
from llama_index.llms.openai import OpenAI as OpenAILlama
from fastapi.middleware.cors import CORSMiddleware
from app_session3 import OpenAIAnswerGenerator, LangChainAnswerGenerator, LlamaIndexAnswerGenerator
import uuid
import redis
import json
from langchain_community.chat_message_histories import RedisChatMessageHistory
from llama_index.storage.chat_store.redis import RedisChatStore
from llama_index.core.memory import ChatMemoryBuffer
import re
from response_structure import SessionRequest, ChatHistoryRequest, QuestionRequest

app = FastAPI()

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
        import ojson
        value = redis_client.get("all_sessions_data_openai")
        value = ojson.loads(value.replace("'", '"'))
        session_id = value[session_name]
        value = redis_client.get(session_id)
        value = re.sub(r"(?<!\w)'(.*?)'(?!\w)", r'"\1"', value)
        value = re.sub(r"\\'", "'", value)
        value = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', value)
        return {"chat_history": ojson.loads(value), "session_id": session_id}
    
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
    
    else:
        return {"error": "Invalid model type"}
    
@app.post("/generate_answer")
async def generate_answer(request: QuestionRequest):
    """
    Endpoint to generate answer from the given question using the specified model type.
    """
    question = request.question
    session_id = request.session_id
    model_type = request.model_type.lower()

    if model_type == "openai":
        model = model_openai
    elif model_type == "langchain":
        model = model_langchain
    elif model_type == "llamaindex":
        model = model_llama
    else:
        raise HTTPException(status_code=400, detail="Invalid model_type. Choose from 'openai', 'langchain', or 'llamaindex'.")

    # Generate and return the answer
    answer = model.generate_answer(question, session_id)
    return {"answer": answer}