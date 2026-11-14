"""Pydantic schemas for LLM classification with instructor."""

from typing import Literal
from pydantic import BaseModel, Field, field_validator


class ItemAnalysisResult(BaseModel):
    item_id: int = Field(description="The unique integer ID of the item being analyzed.")
    relevant: bool = Field(description="True if the text contains meaningful consumer opinion or product discussion.")
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Sentiment towards the product: positive, negative, or neutral."
    )
    score: float = Field(
        ge=-1.0,
        le=1.0,
        description="Sentiment score ranging from -1.0 (extremely negative) to 1.0 (extremely positive).",
    )
    aspect: Literal[
        "price", "quality", "design", "comfort", "brand", "availability", "other"
    ] = Field(description="Primary product aspect discussed.")
    reason: str = Field(description="Brief explanation of the rating (max 20 words).")

    @field_validator("reason")
    @classmethod
    def validate_reason_length(cls, v: str) -> str:
        words = v.strip().split()
        if len(words) > 20:
            raise ValueError(f"Reason must be 20 words or fewer, got {len(words)} words.")
        return v


class BatchAnalysisResult(BaseModel):
    analyses: list[ItemAnalysisResult] = Field(
        description="List of analysis results for each analyzed item in the batch."
    )
