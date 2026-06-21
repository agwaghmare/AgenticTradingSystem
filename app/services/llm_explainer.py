"""
Mistral version of the LLM explainer layer.

Read-only convenience layer — never places orders or influences trading.
"""

import json
import logging

from app.config import settings

logger = logging.getLogger("llm_explainer")

try:
    from mistralai import Mistral

    client = Mistral(api_key=settings.mistral_api_key) if settings.mistral_api_key else None
except ImportError:
    client = None

MODEL = "mistral-small-latest"


def _llm_unavailable() -> str:
    return "LLM explainer unavailable: set MISTRAL_API_KEY in .env to enable narration."


def narrate_cycle(decisions: list[dict], risk_snapshot: dict) -> str:
    if client is None:
        return _llm_unavailable()

    prompt = f"""You are summarizing one trading cycle for a human reviewing
a systematic pairs-trading agent. You are NOT making trading decisions —
the decisions below have already been made and executed by deterministic
code. Your only job is to summarize clearly and flag anything that seems
worth a human's attention.

Risk snapshot for this cycle:
{json.dumps(risk_snapshot, default=str, indent=2)}

Decisions made this cycle:
{json.dumps(decisions, default=str, indent=2)}

Write a concise summary (under 150 words):
- What was traded, if anything, and why (in plain English, not jargon dump)
- What was rejected and why, grouped by reason if there are repeats
- Anything that looks anomalous or worth a human double-checking
  (e.g. unusual rejection patterns, regime/drawdown halts, VaR near cap)

Do not recommend trades. Do not second-guess the risk engine's rules. Just
report what happened and surface anything noteworthy."""

    response = client.chat.complete(
        model=MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return _extract_text(response)


def review_decisions(decisions: list[dict], window_label: str = "last 24h") -> str:
    if client is None:
        return _llm_unavailable()

    prompt = f"""You are reviewing a batch of trading-agent decision logs from
the {window_label} for a systematic pairs-trading desk. These decisions were
made by deterministic risk-rule code, not by you. Your job is purely
analytical review — identify patterns a human risk officer would want to
know about.

Decision logs:
{json.dumps(decisions, default=str, indent=2)}

Look for:
- Repeated rejections on the same pair (possible broken pair, stale config)
- Clusters of rejections by a single reason (possible systemic issue, e.g.
  a data feed problem causing stale-price rejections)
- Any HALTED or EXITS_ONLY periods and how long they lasted
- Unusual frequency of trades vs rejections relative to the rest of the batch

Output a short bulleted list of findings. If nothing stands out, say so
plainly rather than manufacturing a concern. Do not suggest changing risk
thresholds — flag for human review only."""

    response = client.chat.complete(
        model=MODEL,
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    return _extract_text(response)


def _extract_text(response) -> str:
    return response.choices[0].message.content.strip()
