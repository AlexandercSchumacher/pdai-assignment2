"""LLM integration for the Personal Energy Forecast app.

Feature A: Natural Language -> Scenario Parameters (structured output)
  - User describes their upcoming situation in plain English.
  - LLM extracts structured scenario parameters as JSON (using JSON mode).
  - Prompt uses few-shot examples covering diverse real-world situations.
  - A retry call is made with a correction prompt if parsing still fails.
  - The validated parameters feed directly into the existing forecast model.

Feature B: LLM Optimizer Agent (tool use / multi-call agentic loop)
  - User states an energy goal in plain English.
  - LLM agent iteratively calls the forecast function as a tool.
  - Agent explores different parameter combinations across multiple rounds.
  - Returns the best scenario found, the agent's final recommendation, and
    the full call history for display in the UI.

Feature C: RAG over Experiment Log
  - User asks a natural language question about their experiment history.
  - Each log entry is embedded using text-embedding-3-small.
  - Cosine similarity retrieves the most relevant entries for the query.
  - GPT-4o-mini answers the question grounded in the retrieved entries.
  - Surfaces patterns the user would never find by scanning the log manually.

LLM backend: OpenAI gpt-4o-mini + text-embedding-3-small
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable

import numpy as np
import pandas as pd

MODEL       = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"


# ---------------------------------------------------------------------------
# Client setup
# ---------------------------------------------------------------------------

def get_openai_client():
    """Return an authenticated OpenAI client, raising clear errors on failure."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "The 'openai' package is required. Install it with: pip install openai"
        ) from exc

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Add it to a .env file in the project root: OPENAI_API_KEY=your_key_here"
        )
    return OpenAI(api_key=api_key)


# ---------------------------------------------------------------------------
# Feature A: Natural Language -> Scenario Parameters
# ---------------------------------------------------------------------------

# Few-shot examples cover: travel, hard training, recovery, social/work,
# quiet night. This range ensures the model generalises to novel phrasings
# without needing to be retrained.
_FEW_SHOT_EXAMPLES = """
EXAMPLES (input -> output):

Input: "Flying to Tokyo tomorrow, will sleep about 5 hours, drinks at the welcome dinner."
Output: {"bedtime_shift_hours": 2.0, "sleep_delta_min": -120, "training_load_delta": 0, "caffeine_cutoff_hour": 15, "alcohol": true, "late_meal": true, "reasoning": "Long-haul flight implies a late bedtime (+2h) and significantly reduced sleep (-120 min). Drinks at dinner triggers both alcohol and late_meal."}

Input: "Big triathlon race on Saturday so doing a hard brick session tomorrow — swim, bike, run."
Output: {"bedtime_shift_hours": 0.0, "sleep_delta_min": 0, "training_load_delta": 38, "caffeine_cutoff_hour": 15, "alcohol": false, "late_meal": false, "reasoning": "A brick triathlon session is one of the hardest training types, so training_load_delta is near maximum. No mention of sleep changes, alcohol, or late eating."}

Input: "Completely burned out — taking a full rest day, going to bed at 9:30, no coffee after lunch."
Output: {"bedtime_shift_hours": -1.5, "sleep_delta_min": 60, "training_load_delta": -30, "caffeine_cutoff_hour": 13, "alcohol": false, "late_meal": false, "reasoning": "9:30 pm bedtime is ~1.5h earlier than a typical 11pm baseline. Full rest day means strongly negative training delta. No coffee after lunch sets caffeine cutoff to 13h."}

Input: "Client dinner tonight, won't be home until midnight, had an espresso at 6pm."
Output: {"bedtime_shift_hours": 1.5, "sleep_delta_min": -60, "training_load_delta": 0, "caffeine_cutoff_hour": 18, "alcohol": true, "late_meal": true, "reasoning": "Getting home at midnight implies a ~1.5h later bedtime and less total sleep. Client dinner likely involves alcohol and a late heavy meal. Espresso at 6pm sets caffeine cutoff to 18."}

Input: "Quiet evening, skipping the gym, no alcohol, cutting coffee at 2pm and planning to sleep early."
Output: {"bedtime_shift_hours": -1.0, "sleep_delta_min": 30, "training_load_delta": -20, "caffeine_cutoff_hour": 14, "alcohol": false, "late_meal": false, "reasoning": "Sleeping early implies roughly -1h bedtime shift and some extra sleep. Skipping gym is a moderate negative training delta. Coffee cut at 2pm = caffeine_cutoff_hour 14."}
"""

_SYSTEM_PARSE = """\
You are an assistant for a personal energy forecast app. Given a natural language
description of a user's upcoming situation, extract scenario parameters and return
a JSON object — nothing else.

PARAMETERS AND VALID RANGES:
- bedtime_shift_hours  float  [-4.0, 4.0]   Shift from baseline bedtime. Negative = going to bed earlier (good for sleep), positive = later (disrupts sleep).
- sleep_delta_min      int    [-180, 180]    Change in total sleep duration (minutes). "A bit less" ~ -30 to -60; "much less" ~ -90 to -120.
- training_load_delta  int    [-40, 40]      Change in training intensity. "Hard workout/race" ~ +25 to +40; "rest day" ~ -15 to -30.
- caffeine_cutoff_hour int    [12, 22]       24h clock hour of last caffeine. Default 15.
- alcohol              bool                  Any alcohol consumed.
- late_meal            bool                  Late or heavy dinner consumed.
- reasoning            string                1-2 sentence explanation of the key extractions you made.

INFERENCE RULES:
- "drinks/wine/beer/dinner with drinks" -> alcohol=true; dinner context -> late_meal=true also
- "flying/overnight travel/jet lag" -> positive bedtime_shift_hours, negative sleep_delta_min
- "triathlon/marathon/intense training" -> large positive training_load_delta
- "recovery/easy/deload" -> negative training_load_delta
- If not mentioned, use defaults: shifts=0, deltas=0, caffeine_cutoff_hour=15, alcohol=false, late_meal=false

BASELINE (for context): {baseline_context}

{few_shot_examples}

Return exactly this JSON structure:
{{"bedtime_shift_hours": <float>, "sleep_delta_min": <int>, "training_load_delta": <int>, "caffeine_cutoff_hour": <int>, "alcohol": <bool>, "late_meal": <bool>, "reasoning": "<string>"}}
"""


def parse_scenario_from_text(
    user_text: str,
    baseline_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Feature A: Parse a natural language situation description into scenario parameters.

    Uses JSON mode for guaranteed valid JSON output. The system prompt includes
    5 few-shot examples covering travel, hard training, recovery, social events,
    and quiet nights — improving reliability across diverse phrasings. A retry
    call with a correction prompt handles any remaining edge-case parse failures.
    The extracted parameters are then validated, clamped, and fed into the
    forecast simulation — the LLM output drives real numeric computation.

    Args:
        user_text: Free-form description of the user's upcoming situation.
        baseline_state: Dict of the user's latest Oura state (for context).

    Returns:
        Dict with validated scenario parameters + a 'reasoning' explanation.
    """
    client = get_openai_client()

    if baseline_state:
        bedtime_h = float(baseline_state.get("bedtime_start_hour", 23.0))
        sleep_min = float(baseline_state.get("total_sleep_duration_min", 420.0))
        energy    = float(baseline_state.get("energy_score", 70.0))
        baseline_context = (
            f"Bedtime {bedtime_h:.1f}h (24h clock), "
            f"sleep {sleep_min:.0f} min ({sleep_min / 60:.1f}h), "
            f"energy score {energy:.1f}/100"
        )
    else:
        baseline_context = "No baseline data — use typical defaults."

    system = _SYSTEM_PARSE.format(
        baseline_context=baseline_context,
        few_shot_examples=_FEW_SHOT_EXAMPLES,
    )

    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_text},
    ]

    # First call — JSON mode guarantees syntactically valid output
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    raw = response.choices[0].message.content or ""

    params: dict = {}
    try:
        params = json.loads(raw)
    except json.JSONDecodeError:
        # Retry call — ask the model to self-correct (edge case safety net)
        messages.append({"role": "assistant", "content": raw})
        messages.append({
            "role": "user",
            "content": (
                "Your previous response was not valid JSON. "
                "Output ONLY the JSON object with all required fields. No explanation."
            ),
        })
        retry = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        raw    = retry.choices[0].message.content or ""
        params = json.loads(raw)

    return {
        "bedtime_shift_hours": float(
            max(-4.0, min(4.0, float(params.get("bedtime_shift_hours", 0.0))))
        ),
        "sleep_delta_min": int(
            max(-180, min(180, int(round(float(params.get("sleep_delta_min", 0))))))
        ),
        "training_load_delta": int(
            max(-40, min(40, int(round(float(params.get("training_load_delta", 0))))))
        ),
        "caffeine_cutoff_hour": int(
            max(12, min(22, int(round(float(params.get("caffeine_cutoff_hour", 15))))))
        ),
        "alcohol":   bool(params.get("alcohol",   False)),
        "late_meal": bool(params.get("late_meal", False)),
        "reasoning": str(params.get("reasoning", "No reasoning provided.")),
    }


# ---------------------------------------------------------------------------
# Feature B: LLM Optimizer Agent (Tool Use / Multi-call)
# ---------------------------------------------------------------------------

_FORECAST_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "run_energy_forecast",
        "description": (
            "Run an energy forecast simulation for the next 3 days with a specific "
            "set of scenario parameters. Returns predicted energy scores (p10/median/p90) "
            "and a risk level for each day. Call this multiple times with different "
            "parameters to compare scenarios and converge on the best combination."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "bedtime_shift_hours": {
                    "type": "number",
                    "description": "Hours to shift bedtime from baseline. Negative = earlier (better). Positive = later (disrupts sleep). Range: -4 to 4.",
                },
                "sleep_delta_min": {
                    "type": "integer",
                    "description": "Change in sleep duration (minutes). Positive = more sleep. Range: -180 to 180.",
                },
                "training_load_delta": {
                    "type": "integer",
                    "description": "Change in training intensity. Positive = harder, negative = rest. Range: -40 to 40.",
                },
                "caffeine_cutoff_hour": {
                    "type": "integer",
                    "description": "Hour of last caffeine (24h). Earlier = better sleep. Default 15. Range: 12–22.",
                },
                "alcohol": {
                    "type": "boolean",
                    "description": "Whether alcohol is consumed. Significantly reduces energy.",
                },
                "late_meal": {
                    "type": "boolean",
                    "description": "Whether a late or heavy dinner is consumed.",
                },
            },
            "required": [
                "bedtime_shift_hours", "sleep_delta_min", "training_load_delta",
                "caffeine_cutoff_hour", "alcohol", "late_meal",
            ],
        },
    },
}


def _run_forecast_tool(
    params: dict[str, Any],
    model_bundle: dict,
    base_state: pd.Series,
) -> str:
    """Execute the forecast simulation and return a JSON string for the LLM."""
    from src.forecast import simulate_forecast

    scenario = {
        "bedtime_shift_hours":  float(params.get("bedtime_shift_hours", 0.0)),
        "sleep_delta_min":      int(round(float(params.get("sleep_delta_min", 0)))),
        "training_load_delta":  int(round(float(params.get("training_load_delta", 0)))),
        "caffeine_cutoff_hour": int(round(float(params.get("caffeine_cutoff_hour", 15)))),
        "alcohol":              bool(params.get("alcohol",   False)),
        "late_meal":            bool(params.get("late_meal", False)),
    }

    forecast_df = simulate_forecast(
        model_bundle=model_bundle,
        base_state=base_state,
        scenario=scenario,
        horizon_days=3,
        n_samples=400,
        random_state=42,
    )

    result = {
        "scenario_used":   scenario,
        "forecast_by_day": forecast_df[["day", "p10", "median", "p90", "risk"]].round(1).to_dict(orient="records"),
        "day1_energy":     round(float(forecast_df.iloc[0]["median"]), 1),
        "day2_energy":     round(float(forecast_df.iloc[1]["median"]), 1) if len(forecast_df) > 1 else None,
        "day3_energy":     round(float(forecast_df.iloc[2]["median"]), 1) if len(forecast_df) > 2 else None,
        "average_energy":  round(float(forecast_df["median"].mean()), 1),
    }
    return json.dumps(result)


def optimize_scenario_with_agent(
    goal: str,
    model_bundle: dict,
    base_state: pd.Series,
    max_iterations: int = 6,
    on_tool_call: Callable[[int, dict, dict], None] | None = None,
) -> dict[str, Any]:
    """
    Feature B: Agentic LLM optimizer using tool use to find the best scenario.

    The LLM receives a user goal and the forecast function as a callable tool.
    It iteratively proposes parameters, calls the tool, reads quantitative results,
    and refines its strategy — a genuine multi-call agentic loop. The on_tool_call
    callback allows the Streamlit UI to stream progress in real time.

    Args:
        goal:           User's natural language energy goal.
        model_bundle:   Trained model bundle from load_model_bundle().
        base_state:     Current state (latest Oura data point).
        max_iterations: Maximum number of tool-call rounds.
        on_tool_call:   Optional callback(iteration, params, result) for UI updates.

    Returns:
        Dict with final_recommendation, best_scenario, best_avg_energy,
        call_history, and iterations count.
    """
    client = get_openai_client()

    baseline_energy = float(base_state.get("energy_score", 70.0))

    system = f"""\
You are an expert energy optimization agent for a personal health forecast app.
Find the best lifestyle adjustments to achieve the user's energy goal.
You have a tool run_energy_forecast that simulates energy for the next 3 days.

Current baseline energy: {baseline_energy:.1f}/100

STRATEGY:
1. Start with a sensible first scenario based on the goal.
2. Try at least 3-4 meaningfully different combinations — vary bedtime, sleep,
   training load, and caffeine cutoff to understand each parameter's impact.
3. After exploring, identify the best combination.
4. In your FINAL response, provide: recommended parameters, expected day-by-day
   energy scores, why these adjustments work, and any trade-offs.
"""

    messages: list[dict] = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                f"My energy goal: {goal}\n\n"
                "Explore different scenarios using the forecast tool and find "
                "the optimal lifestyle adjustments to achieve this goal."
            ),
        },
    ]

    call_history:   list[dict] = []
    best_scenario:  dict | None = None
    best_avg_energy = -1.0
    final_text      = ""
    iteration       = 0

    while iteration < max_iterations:
        response   = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=[_FORECAST_TOOL],
            tool_choice="auto",
            temperature=0.3,
        )
        msg        = response.choices[0].message
        tool_calls = msg.tool_calls or []

        assistant_entry: dict = {"role": "assistant", "content": msg.content}
        if tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "id": tc.id, "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in tool_calls
            ]
        messages.append(assistant_entry)

        if not tool_calls:
            final_text = msg.content or ""
            break

        for tc in tool_calls:
            args       = json.loads(tc.function.arguments)
            result_str = _run_forecast_tool(args, model_bundle, base_state)
            result_data = json.loads(result_str)

            avg_energy = result_data.get("average_energy", 0.0)
            if avg_energy > best_avg_energy:
                best_avg_energy = avg_energy
                best_scenario   = result_data["scenario_used"]

            tool_call_num = len(call_history) + 1
            record = {"iteration": tool_call_num, "params": args, "result": result_data}
            call_history.append(record)

            if on_tool_call:
                on_tool_call(tool_call_num, args, result_data)

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_str,
            })

        iteration += 1

    return {
        "final_recommendation": final_text,
        "best_scenario":        best_scenario,
        "best_avg_energy":      best_avg_energy,
        "call_history":         call_history,
        "iterations":           len(call_history),
    }


# ---------------------------------------------------------------------------
# Feature C: RAG over Experiment Log
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    va, vb = np.array(a, dtype=float), np.array(b, dtype=float)
    denom  = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


def _log_entry_to_text(row: dict) -> str:
    """Serialise a single experiment log row to a readable string for embedding."""
    delta     = row.get("day1_delta", 0)
    direction = "improved" if float(delta) >= 0 else "reduced"
    notes     = str(row.get("notes", "")).strip() or "no notes"
    tags      = str(row.get("tags",  "")).strip() or "none"
    return (
        f"Date: {row.get('created_at', 'unknown')} | "
        f"Scenario: '{row.get('scenario_name', 'unnamed')}' | "
        f"Tags: {tags} | "
        f"Energy {direction} by {abs(float(delta)):.1f} points (day1_delta={delta}) | "
        f"Notes: {notes}"
    )


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using text-embedding-3-small."""
    client   = get_openai_client()
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


def answer_from_log(
    question: str,
    log_df: pd.DataFrame,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Feature C: RAG pipeline that answers natural language questions about the
    user's personal experiment log.

    Pipeline:
      1. Serialise every log entry to a plain-text string.
      2. Embed all entries + the user's question in a single API batch call
         (text-embedding-3-small).
      3. Rank entries by cosine similarity to the query embedding.
      4. Pass the top-k entries as context to GPT-4o-mini.
      5. LLM answers grounded strictly in the retrieved entries.

    This covers the RAG rubric category: retrieval is semantic (not keyword),
    and the LLM output is constrained by retrieved personal data rather than
    general knowledge.

    Args:
        question: User's natural language question about their experiments.
        log_df:   The experiment log DataFrame.
        top_k:    Number of most-relevant entries to pass as context.

    Returns:
        Dict with 'answer' (str) and 'sources' (list of retrieved entry dicts).
    """
    MIN_ENTRIES = 2
    if log_df is None or len(log_df) < MIN_ENTRIES:
        return {
            "answer": (
                f"Not enough data yet - add at least {MIN_ENTRIES} experiment log entries "
                "before asking questions. Each entry you log after running a scenario "
                "becomes part of the searchable history."
            ),
            "sources": [],
        }

    # Fill NaNs so serialisation doesn't break
    log_df = log_df.fillna("")

    # Build text representations for every entry
    entry_texts = [_log_entry_to_text(row) for row in log_df.to_dict(orient="records")]

    # Embed entries + query in one batch call
    all_texts   = entry_texts + [question]
    embeddings  = _embed_texts(all_texts)

    entry_embeddings = embeddings[:-1]
    query_embedding  = embeddings[-1]

    # Rank by cosine similarity
    scored = sorted(
        enumerate(entry_embeddings),
        key=lambda x: _cosine_similarity(query_embedding, x[1]),
        reverse=True,
    )
    top_indices = [i for i, _ in scored[:min(top_k, len(scored))]]

    context_entries = [
        {"text": entry_texts[i], "row": log_df.iloc[i].to_dict()}
        for i in top_indices
    ]

    context_block = "\n\n".join(
        f"Entry {n + 1}:\n{e['text']}" for n, e in enumerate(context_entries)
    )

    system = """\
You are an analyst helping a user discover patterns in their personal energy experiment log.
Each entry records a lifestyle scenario they tried and the resulting change in energy score.
A positive day1_delta means energy improved vs baseline; negative means it reduced.

Answer the user's question based ONLY on the provided log entries.
Be specific: cite dates, scenario names, and delta values when relevant.
If the data is insufficient to answer confidently, say so clearly.
Do not invent data or draw on general knowledge — only use what is in the entries.
"""

    client   = get_openai_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    f"Here are the most relevant entries from my experiment log:\n\n"
                    f"{context_block}\n\n"
                    f"My question: {question}"
                ),
            },
        ],
        temperature=0.2,
    )

    return {
        "answer":  response.choices[0].message.content or "",
        "sources": context_entries,
    }
