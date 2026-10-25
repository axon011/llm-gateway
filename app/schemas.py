from typing import Literal

from pydantic import BaseModel, Field

# Request size caps. Every token is paid for, so an unbounded body is a cost hole once the
# gateway is on the internet. Generous for a tutor chat (long system prompt + history).
MAX_MESSAGE_CHARS = 32_000
MAX_MESSAGES = 200


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)


class GenerationOptions(BaseModel):
    """Per-request sampling. Unset fields keep the provider's defaults."""

    model: str | None = Field(default=None, description="Overrides the configured model for this call")
    temperature: float | None = Field(default=None, ge=0, le=2)
    reasoning_effort: Literal["low", "medium", "high"] | None = Field(
        default=None, description="Groq gpt-oss only; ignored by other providers"
    )


class ChatRequest(GenerationOptions):
    messages: list[Message] = Field(min_length=1, max_length=MAX_MESSAGES)
    session_id: str | None = Field(default=None, description="Groups turns of one conversation in Langfuse")
    user_id: str | None = Field(default=None, description="Hashed learner id; never a raw email")
    route: str = Field(default="default", description="Caller-defined label, e.g. tutor.reply, tutor.correct")


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cost_usd: float | None = Field(description="Tokens x dated list price; null if the model has no price entry")


class ChatResponse(BaseModel):
    text: str
    model: str
    provider: str
    usage: Usage
    trace_id: str | None = None
