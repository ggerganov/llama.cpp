from typing import Any, Optional
from pydantic import Json

from examples.openai.api import LlamaCppParams

class LlamaCppServerCompletionRequest(LlamaCppParams):
    prompt: str
    stream: Optional[bool] = None
    cache_prompt: Optional[bool] = None

    grammar: Optional[str] = None
    json_schema: Optional[Json[Any]] = None
