"""LLM Classifier using LiteLLM + Instructor for structured analysis."""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlmodel import Session, select

from db import get_engine
from models import Analysis, Item
from settings import settings
from analysis.schemas import BatchAnalysisResult, ItemAnalysisResult

# Ensure LiteLLM uses local backup cost map to avoid blocking network calls on startup
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"
DEFAULT_PROMPT_VERSION = "v1"
DEFAULT_BATCH_SIZE = 10
FALLBACK_MODEL = "gemini/gemini-1.5-flash"


def load_prompt_template(prompt_version: str = DEFAULT_PROMPT_VERSION) -> str:
    """Load prompt template from analysis/prompts/{prompt_version}.txt."""
    prompt_file = PROMPTS_DIR / f"{prompt_version}.txt"
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt template file not found: {prompt_file}")
    with open(prompt_file, encoding="utf-8") as f:
        return f.read()


def get_instructor_client() -> Any:
    """Create and return an Instructor client wrapping LiteLLM."""
    import litellm
    import instructor

    litellm.telemetry = False
    litellm.suppress_debug_info = True

    return instructor.from_litellm(litellm.completion)


def analyze_items(
    items: list[Item],
    product: str,
    model: str | None = None,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    session: Session | None = None,
    client: Any | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[Analysis]:
    """Analyze a list of Items for sentiment and aspect toward the specified product.

    - Uses litellm + instructor with temperature=0.
    - Batches ~10 items per call.
    - Skips items that already have an Analysis row for (model, prompt_version).
    - Stores Analysis records in the database.

    Args:
        items: List of Item instances to analyze.
        product: The target product name.
        model: LLM model identifier (defaults to settings.llm_model or FALLBACK_MODEL).
        prompt_version: The versioned prompt name in analysis/prompts/ (default "v1").
        session: Optional SQLModel Session; creates one from db.get_engine() if None.
        client: Optional instructor client (useful for testing with mocks).
        batch_size: Number of items per LLM call (default 10).

    Returns:
        list[Analysis]: All analysis records for the input items.
    """
    if not items:
        return []

    model_name = (model or settings.llm_model or FALLBACK_MODEL).strip()
    template = load_prompt_template(prompt_version)
    system_prompt = template.replace("{product}", product)

    own_session = session is None
    if own_session:
        session = Session(get_engine())

    try:
        # Determine which items are already analyzed for (model_name, prompt_version)
        item_ids = [item.id for item in items if item.id is not None]
        existing_map: dict[int, Analysis] = {}

        if item_ids:
            stmt = select(Analysis).where(
                Analysis.model == model_name,
                Analysis.prompt_version == prompt_version,
                Analysis.item_id.in_(item_ids),
            )
            for existing in session.exec(stmt).all():
                existing_map[existing.item_id] = existing

        # Filter items that need analysis
        unprocessed_items = [
            item for item in items if item.id is not None and item.id not in existing_map
        ]

        # If all items already analyzed, make zero model calls!
        if not unprocessed_items:
            logger.info(
                "All %d items already have analysis for (%s, %s); zero model calls made.",
                len(items),
                model_name,
                prompt_version,
            )
            return [existing_map[item.id] for item in items if item.id in existing_map]

        if client is None:
            client = get_instructor_client()

        call_kwargs: dict[str, Any] = {
            "model": model_name,
            "response_model": BatchAnalysisResult,
            "temperature": 0,
            "max_retries": 2,
        }
        if settings.llm_api_key:
            call_kwargs["api_key"] = settings.llm_api_key

        now_str = datetime.now(timezone.utc).isoformat()
        new_analyses: list[Analysis] = []

        # Process in batches of ~10 items
        for i in range(0, len(unprocessed_items), batch_size):
            batch = unprocessed_items[i : i + batch_size]
            batch_lines = [
                f"Analyze the following {len(batch)} items regarding '{product}':\n"
            ]
            for item in batch:
                batch_lines.append(
                    f"[Item ID: {item.id}]\n"
                    f"Title: {item.title}\n"
                    f"Text: {item.text[:1000]}\n"
                    f"---"
                )
            batch_content = "\n".join(batch_lines)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": batch_content},
            ]

            # Instructor call
            if hasattr(client, "chat") and hasattr(client.chat, "completions"):
                response: BatchAnalysisResult = client.chat.completions.create(
                    messages=messages, **call_kwargs
                )
            elif hasattr(client, "create"):
                response = client.create(messages=messages, **call_kwargs)
            else:
                response = client(messages=messages, **call_kwargs)

            # Map responses back to items
            resp_by_id = {res.item_id: res for res in response.analyses}

            for item in batch:
                assert item.id is not None
                res: ItemAnalysisResult | None = resp_by_id.get(item.id)
                if res is None:
                    # Fallback if model missed an item ID in the batch
                    logger.warning("Item %d missing in model response batch; creating neutral fallback", item.id)
                    analysis = Analysis(
                        item_id=item.id,
                        model=model_name,
                        prompt_version=prompt_version,
                        relevant=False,
                        sentiment="neutral",
                        score=0.0,
                        aspect="other",
                        reason="Missing from batch response",
                        analysed_at=now_str,
                    )
                else:
                    analysis = Analysis(
                        item_id=item.id,
                        model=model_name,
                        prompt_version=prompt_version,
                        relevant=res.relevant,
                        sentiment=res.sentiment,
                        score=res.score,
                        aspect=res.aspect,
                        reason=res.reason,
                        analysed_at=now_str,
                    )

                session.add(analysis)
                new_analyses.append(analysis)
                existing_map[item.id] = analysis

            session.commit()

        return [existing_map[item.id] for item in items if item.id in existing_map]

    finally:
        if own_session:
            session.close()
