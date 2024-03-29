from abc import ABC
from typing import Any, Dict, Literal, Optional, Union
from pydantic import BaseModel, Json, TypeAdapter

class FunctionCall(BaseModel):
    name: str
    arguments: Union[Dict[str, Any], str]

class ToolCall(BaseModel):
    id: Optional[str] = None
    type: Literal["function"] = "function"
    function: FunctionCall

class Message(BaseModel):
    role: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    content: Optional[str]
    tool_calls: Optional[list[ToolCall]] = None

class ToolFunction(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]

class Tool(BaseModel):
    type: str
    function: ToolFunction

class ResponseFormat(BaseModel):
    type: str
    json_schema: Optional[Any] = None

class LlamaCppParams(BaseModel):
    n_predict: Optional[int] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    min_p: Optional[float] = None
    tfs_z: Optional[float] = None
    typical_p: Optional[float] = None
    temperature: float = 1.0
    dynatemp_range: Optional[float] = None
    dynatemp_exponent: Optional[float] = None
    repeat_last_n: Optional[int] = None
    repeat_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presense_penalty: Optional[float] = None
    mirostat: Optional[bool] = None
    mirostat_tau: Optional[float] = None
    mirostat_eta: Optional[float] = None
    penalize_nl: Optional[bool] = None
    n_keep: Optional[int] = None
    seed: Optional[int] = None
    n_probs: Optional[int] = None
    min_keep: Optional[int] = None

class ChatCompletionRequest(LlamaCppParams):
    model: str
    tools: Optional[list[Tool]] = None
    messages: list[Message] = None
    prompt: Optional[str] = None
    response_format: Optional[ResponseFormat] = None

    stream: bool = False
    cache_prompt: Optional[bool] = None

class Choice(BaseModel):
    index: int
    message: Message
    logprobs: Optional[Json] = None
    finish_reason: Union[Literal["stop"], Literal["tool_calls"]]

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class CompletionError(BaseModel):
    message: str
    # code: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: list[Choice]
    usage: Usage
    system_fingerprint: str
    error: Optional[CompletionError] = None