"""Optional second backend: local Hugging Face model for sentiment only.

Uses 'cardiffnlp/twitter-roberta-base-sentiment-latest'.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

HF_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"


class HFClassifier:
    """Optional Hugging Face sentiment classifier pipeline."""

    def __init__(self, model_name: str = HF_MODEL_NAME):
        self.model_name = model_name
        self._pipeline: Any = None

    @property
    def pipeline(self) -> Any:
        if self._pipeline is None:
            try:
                from transformers import pipeline  # lazy import
                self._pipeline = pipeline("text-classification", model=self.model_name)
            except ImportError:
                logger.error(
                    "transformers and torch are not installed. "
                    "Install them to use the Hugging Face sentiment backend."
                )
                raise
        return self._pipeline

    def classify_sentiment(self, text: str) -> tuple[str, float]:
        """Classify sentiment of text using local HF pipeline.

        Returns:
            tuple[str, float]: (sentiment, score) where sentiment is
            'positive', 'negative', or 'neutral', and score is in [-1.0, 1.0].
        """
        truncated = text[:512]
        pred = self.pipeline(truncated)[0]
        raw_label = pred["label"].lower()
        confidence = float(pred["score"])

        if "pos" in raw_label:
            sentiment = "positive"
            score = round(confidence, 4)
        elif "neg" in raw_label:
            sentiment = "negative"
            score = round(-confidence, 4)
        else:
            sentiment = "neutral"
            score = 0.0

        return sentiment, score
