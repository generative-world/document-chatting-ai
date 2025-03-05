
from collections import deque
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from app.retriever import create_retriever
from app.config import LLM_MODEL_NAME, USE_GPU

# Simple in-memory chat history manager
class ChatHistory:
    def __init__(self, max_history=5):
        self.history = deque(maxlen=max_history)
    
    def add_message(self, user_message, bot_message):
        self.history.append({"user": user_message, "bot": bot_message})
    
    def get_history(self):
        return [msg for msg in self.history]

# Initialize conversational chain with retrieval
def setup_conversational_chain():
    retriever = create_retriever()
    
    llm = OpenAI(model=LLM_MODEL_NAME, temperature=0.7) if not USE_GPU else OpenAI(model=LLM_MODEL_NAME, temperature=0.7, device='cuda')
    
    conversation_chain = ConversationalRetrievalChain.from_llm(llm, retriever)
    return conversation_chain

def get_response(conversation_chain, query, chat_history):
    # Fetch the response from the model
    response = conversation_chain.run(query)
    
    # Add to chat history
    chat_history.add_message(query, response)
    
    return response
