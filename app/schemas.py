from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1)
    session_id: str | None = Field(default=None, description="Groups turns of one conversation in Langfuse")
    user_id: str | None = Field(default=None, description="Hashed learner id; never a raw email")
    route: str = Field(default="default", description="Caller-defined label, e.g. tutor.reply, tutor.correct")


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class ChatResponse(BaseModel):
    text: str
    model: str
    provider: str
    usage: Usage
    trace_id: str | None = None
