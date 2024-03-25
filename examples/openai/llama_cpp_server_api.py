from typing import Optional
from pydantic import BaseModel, Json

class LlamaCppServerCompletionRequest(BaseModel):
    prompt: str
    stream: Optional[bool] = None
    cache_prompt: Optional[bool] = None
    n_predict: Optional[int] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    min_p: Optional[float] = None
    tfs_z: Optional[float] = None
    typical_p: Optional[float] = None
    temperature: Optional[float] = None
    dynatemp_range: Optional[float] = None
    dynatemp_exponent: Optional[float] = None
    repeat_last_n: Optional[int] = None
    repeat_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    mirostat: Optional[bool] = None
    mirostat_tau: Optional[float] = None
    mirostat_eta: Optional[float] = None
    penalize_nl: Optional[bool] = None
    n_keep: Optional[int] = None
    seed: Optional[int] = None
    grammar: Optional[str] = None
    json_schema: Optional[Json] = None