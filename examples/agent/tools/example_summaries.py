
from typing import Annotated, List, Optional
from annotated_types import MinLen
from pydantic import BaseModel


class QAPair(BaseModel):
    question: str
    concise_answer: str
    justification: str

class PyramidalSummary(BaseModel):
    title: str
    summary: str
    question_answers: Annotated[List[QAPair], MinLen(2)]
    sub_sections: Optional[Annotated[List['PyramidalSummary'], MinLen(2)]]
