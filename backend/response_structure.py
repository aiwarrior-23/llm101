from pydantic import BaseModel

class QuestionRequest(BaseModel):
    question: str
    model_type: str
    session_id: str
    
class ChatHistoryRequest(BaseModel):
    session_name: str
    model_type: str
    
class SessionRequest(BaseModel):
    model_type: str