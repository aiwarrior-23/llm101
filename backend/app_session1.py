from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI as OpenAILlama
from fastapi.middleware.cors import CORSMiddleware

from config import OPENAI_API_KEY, MODEL
from prompts import SYSTEM_PROMPT

client_openai = OpenAI(api_key=OPENAI_API_KEY)
client_langchain = ChatOpenAI(model=MODEL, api_key=OPENAI_API_KEY)
client_llamaindex = OpenAILlama(model=MODEL, api_key=OPENAI_API_KEY)

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

def generate_answers(question: str):
    """
    Take the user question and generate an answer using the OpenAI model.
    """
    try:
        answer = client_openai.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}
            ]
        )
        return answer.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error generating answer")
    
def generate_answers_langchain(question: str):
    """
    Take the user question and generate an answer using the OpenAI model.
    """
    try:
        answer = client_langchain.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=question)
        ])
        return answer.content
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error generating answer")
    
def generate_answers_llama_index(question: str):
    """
    Take the user question and generate an answer using the OpenAI model.
    """
    try:
        answer = client_llamaindex.chat([
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=question)
        ])
        return str(answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error generating answer")

@app.post("/generate_answer")
async def get_answer(question: Question):
    """
    Endpoint to generate answer from the given question using openai
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
async def get_answer_with_llama_index(question: Question):
    """
    Endpoint to generate answer from the given question using llama index
    """
    answer = generate_answers_llama_index(question.question)
    return {"answer": answer}