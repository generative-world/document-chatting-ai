from fastapi import FastAPI
from pydantic import BaseModel
from app.conversation import setup_conversational_chain, get_response, ChatHistory
from app.config import LLM_MODEL_NAME

app = FastAPI()

# Initialize the conversation chain and chat history
conversation_chain = setup_conversational_chain()
chat_history = ChatHistory()

class QueryRequest(BaseModel):
    query: str

@app.post("/get_response/")
def chat_with_document(request: QueryRequest):
    user_query = request.query
    response = get_response(conversation_chain, user_query, chat_history)
    return {"response": response, "chat_history": chat_history.get_history()}

@app.get("/")
def read_root():
    return {"message": "Welcome to Document-Based Conversational AI!"}
