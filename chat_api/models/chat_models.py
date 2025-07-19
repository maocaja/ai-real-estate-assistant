from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class Message(BaseModel):
    role: str
    content: Optional[str] = None  # <--- Here's the relevant part
    tool_calls: Optional[List[Dict[str, Any]]] = None # Assuming tool_calls are a list of dicts
    tool_call_id: Optional[str] = None # For tool responses
    name: Optional[str] = None # For tool responses
    
class ChatRequest(BaseModel):
    """
    Represents a request to the chat API.
    Contains the current user message and optional conversation history.
    """
    current_message: str
    conversation_history: List[Message] = [] # List of previous messages

class ChatResponse(BaseModel):
    """
    Represents the response from the chat API.
    """
    response_message: str
    # You might want to include other metadata here, like:
    # relevant_project_ids: List[str] = []
    # debug_info: Optional[Dict[str, Any]] = None


class LLMRequest(BaseModel):
    messages: List[Message]
    model: str
    temperature: float = 0.7 # Default temperature for creative output

class LLMResponse(BaseModel):
    response_content: Optional[str] = None 
    tool_calls: Optional[List[Dict[str, Any]]] = None
    # Other potential fields from LLM response, e.g., token usage, finish reason