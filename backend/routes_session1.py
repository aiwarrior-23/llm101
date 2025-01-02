from fastapi import FastAPI
from pydantic import BaseModel
from llama_index.llms.openai import OpenAI as OpenAILlama
from fastapi.middleware.cors import CORSMiddleware
from app_session1 import generate_answers, generate_answers_langchain, generate_answers_llama_index
import uuid
import redis
import json
import re
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    question: str
    model_type: str
    
@app.post("/generate_answer")
async def get_answer(question: Question):
    """
    Endpoint to generate answer from the given question.
    """
    answer = generate_answers(question.question)
    return {"answer": answer}
    
@app.post("/generate_answer_using_langchain")
async def get_answer_with_langchain(question: Question):
    """
    Endpoint to generate answer from the given question using langchain
    """
    answer = generate_answers_langchain(question.question)
    return {"answer": answer}

@app.post("/generate_answer_using_llama_index")
async def get_answer_with_langchain(question: Question):
    """
    Endpoint to generate answer from the given question using llama index
    """
    answer = generate_answers_llama_index(question.question)
    return {"answer": answer}