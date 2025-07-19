from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class Message(BaseModel):
    """Represents a single message in the conversation."""
    role: str # e.g., "user", "assistant", "system"
    content: str

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
    response_content: str
    # Other potential fields from LLM response, e.g., token usage, finish reason