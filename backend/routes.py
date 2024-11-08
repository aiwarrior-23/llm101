from fastapi import FastAPI
from pydantic import BaseModel
from llama_index.llms.openai import OpenAI as OpenAILlama
from fastapi.middleware.cors import CORSMiddleware
from app_session2 import OpenAIAnswerGenerator, LangChainAnswerGenerator, LlamaIndexAnswerGenerator

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

class QuestionRequest(BaseModel):
    question: str
    model_type: str

@app.post("/generate_answer")
async def generate_answer(request: QuestionRequest):
    """
    Endpoint to generate answer from the given question using the specified model type.
    """
    question = request.question
    model_type = request.model_type.lower()
    print(model_type)

    # Select the appropriate model generator based on model_type
    if model_type == "openai":
        model = model_openai
    elif model_type == "langchain":
        model = model_langchain
    elif model_type == "llamaindex":
        model = model_llama
    else:
        raise HTTPException(status_code=400, detail="Invalid model_type. Choose from 'openai', 'langchain', or 'llamaindex'.")

    # Generate and return the answer
    answer = model.generate_answer(question)
    return {"answer": answer}