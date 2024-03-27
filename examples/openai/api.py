from typing import Any, Dict, Literal, Optional, Union
from pydantic import BaseModel, Json

class FunctionCall(BaseModel):
    name: str
    arguments: Dict[str, Any]

class ToolCall(BaseModel):
    id: Optional[str] = None
    type: Literal["function"] = "function"
    function: FunctionCall

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

class Choice(BaseModel):
    index: int
    message: Message
    logprobs: Optional[Json] = None
    finish_reason: Union[Literal["stop"], Literal["tool_calls"]]

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: list[Choice]
    usage: Usage
    system_fingerprint: str