"""Gemini Vision: turn a chart/graph image into searchable text.
Add this function to app/llm.py if you prefer keeping all Gemini code there."""
import os
from typing import Optional

CHART_PROMPT = (
    "You are reading a chart or graph from a company annual report. "
    "Describe ONLY what is visible. Give: chart type, title, axis labels and units, "
    "each series/legend name, the key numeric values (read from labels or estimate and "
    "say 'approx.'), and the main trend. If it is not a chart/graph (logo, photo, "
    "decoration) reply exactly: NOT_A_CHART. Do not add outside knowledge."
)


def describe_image(png: bytes) -> Optional[str]:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return None
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=key)
    resp = client.models.generate_content(
        model=os.getenv("GEMINI_VISION_MODEL", os.getenv("GEMINI_MODEL", "gemini-2.0-flash")),
        contents=[types.Part.from_bytes(data=png, mime_type="image/png"), CHART_PROMPT],
    )
    text = (resp.text or "").strip()
    return None if not text or text.startswith("NOT_A_CHART") else text
