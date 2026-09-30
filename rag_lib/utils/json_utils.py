"""Shared utilities for parsing LLM JSON responses.

Provides a single ``parse_llm_json`` function that handles the common
quirks of LLM-generated JSON: markdown code fences, trailing commas,
Python-style booleans, extra text around the JSON block, etc.

Used by:
- ``pipeline/services/image/image_extractor.py``
- ``pipeline/services/image/image_chunk_generator.py``
- ``win-chat/backend/services/rag_engine/smart_agent.py``
"""
from __future__ import annotations

import json
import re
from typing import Optional


def parse_llm_json(raw: str) -> Optional[dict]:
    """Robustly extract a JSON object from an LLM response.

    Handles:
    - Markdown code fences (````` ```json ... ``` `````)
    - Trailing commas before ``}`` or ``]``
    - Extra text before / after the JSON block
    - Nested braces (string-aware to skip braces inside quoted values)
    - Python-style booleans (``True`` / ``False`` / ``None``)

    Returns:
        Parsed ``dict``, or ``None`` if extraction fails.
    """
    if not raw or not raw.strip():
        return None

    # 1. Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", raw)
    cleaned = cleaned.strip()

    # 2. Find the outermost { ... } using a string-aware bracket-depth scan
    start = cleaned.find("{")
    if start == -1:
        return None

    depth = 0
    end = -1
    in_string = False
    escape_next = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        return None

    raw_json = cleaned[start : end + 1]

    # 3. Progressive parsing with increasingly aggressive fixes
    for candidate in _fix_candidates(raw_json):
        try:
            result = json.loads(candidate)
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            continue

    return None


def _fix_candidates(raw_json: str):
    """Yield progressively more tolerant versions of *raw_json*."""
    # Direct attempt
    yield raw_json

    # Fix trailing commas:  ,} → }  and  ,] → ]
    fixed = re.sub(r",\s*([}\]])", r"\1", raw_json)
    yield fixed

    # Python-style booleans / None → JSON equivalents
    fixed2 = re.sub(r"\bTrue\b", "true", fixed)
    fixed2 = re.sub(r"\bFalse\b", "false", fixed2)
    fixed2 = re.sub(r"\bNone\b", "null", fixed2)
    yield fixed2

    # Single quotes → double quotes (very aggressive, last resort)
    fixed3 = fixed2.replace("'", '"')
    yield fixed3
