from typing import Any, Dict, Optional
from pydantic import BaseModel, Json

class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any]

class Message(BaseModel):
    role: str
    content: Optional[str]
    tool_calls: Optional[list[ToolCall]] = None

class ToolFunction(BaseModel):
    name: str
    description: str
    parameters: Any

class Tool(BaseModel):
    type: str
    function: ToolFunction

class ResponseFormat(BaseModel):
    type: str
    json_schema: Optional[Any] = None

class ChatCompletionRequest(BaseModel):
    model: str
    tools: Optional[list[Tool]] = None
    messages: list[Message]
    response_format: Optional[ResponseFormat] = None
    temperature: float = 1.0
    stream: bool = False
