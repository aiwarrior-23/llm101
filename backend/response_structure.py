from pydantic import BaseModel, Field

class QuestionRequest(BaseModel):
    question: str
    model_type: str
    session_id: str
    
class ChatHistoryRequest(BaseModel):
    session_name: str
    model_type: str
    
class SessionRequest(BaseModel):
    model_type: str

class QueryRequest(BaseModel):
    question: str
    model_type: str
    session_id: str
    data_path: str = Field(None, description="Path to the uploaded data file.")
    database_name: str
    regenerate: bool = True