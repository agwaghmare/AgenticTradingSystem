"""
LLM explainer layer for the trading agent.

IMPORTANT: This module is read-only with respect to trading. It NEVER places
orders, NEVER influences risk checks, and NEVER changes agent behavior. It
only narrates decisions that the deterministic TradingAgent has already made
and logged. Keeping this strictly downstream of execution preserves the
auditability and determinism of the core system.

Two use cases:
1. narrate_cycle()  - turn one cycle's structured decisions into a plain-
   English summary (for a dashboard, Slack message, or daily digest).
2. review_decisions() - batch-review a window of decision_log rows and flag
   anything that looks anomalous, for human review only.
"""

import json
from anthropic import Anthropic

from app.config import settings

client = Anthropic(api_key=settings.anthropic_api_key)

MODEL = "claude-sonnet-4-6"


def narrate_cycle(decisions: list[dict], risk_snapshot: dict) -> str:
    """
    decisions: list of decision_log rows (as dicts) from one cycle_id.
    risk_snapshot: the risk_snapshot row for that same cycle.
    Returns a short plain-English summary suitable for a Slack/email digest.
    """
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

    response = client.messages.create(
        model=MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return _extract_text(response)


def review_decisions(decisions: list[dict], window_label: str = "last 24h") -> str:
    """
    Batch review for anomaly detection. Intended to run periodically
    (e.g. daily cron), not per-cycle. Flags patterns for human review only.
    """
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

    response = client.messages.create(
        model=MODEL,
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    return _extract_text(response)


def _extract_text(response) -> str:
    return "".join(block.text for block in response.content if block.type == "text")
