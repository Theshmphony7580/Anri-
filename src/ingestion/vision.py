import base64
import logging
from typing import Optional

import httpx

from config import get_settings

logger = logging.getLogger(__name__)

# Finalized Forensic Document Intelligence Engine prompt for dense RAG indexing
DEFAULT_CHART_PROMPT = """Role: You are a Forensic Document Intelligence Engine parsing visual figures for an enterprise RAG knowledge base.
Objective: Extract all visible semantic, architectural, and quantitative information with forensic accuracy.

### Execution Format:

1. VISUAL IDENTITY & TITLE:
   - Primary Title: [Transcribe verbatim, including any subtitles or figure numbers]
   - Media Classification: [e.g., System Architecture, Sequence Flow, Bar Graph, Line Chart, Entity-Relationship, Network Topology]

2. LABELED ENTITY INVENTORY:
   - Named Nodes / Components: [Alphabetical list of all labeled boxes, servers, services, database tables, or technologies]
   - Categorical Legends & Keys: [Exact text of all legend labels, color-coded keys, and footnotes]

3. DATA VALUES & QUANTITATIVE FINDINGS (If chart / plot):
   - Axes & Units: [X-axis label, Y-axis label, scale intervals, and measurement units]
   - Discrete Metrics: [Transcribe key data points, high/low extremes, anomalies, and exact percentage changes]

4. STRUCTURAL FLOW & TOPOLOGY (If diagram / flowchart):
   - Directional Chains: [Map flows strictly in format: Source -> Action/Protocol -> Target]
   - Cluster Groupings: [List elements contained inside boundary boxes, VPCs, subnets, or conceptual zones]

5. FORENSIC SYNTHESIS:
   - Primary Takeaway: [Max 25 words: the core technical fact or business metric demonstrated]

### Negative Constraints (Strict):
- Never output conversational preamble, disclaimers, or introductory sentences.
- Never guess or extrapolate: if a label or number is partially cropped or illegible, write "[illegible]".
- Never comment on visual aesthetics, colors, or diagram quality.
- Output raw structured markdown only."""


class VLMVisionCaptioner:
    """Generates forensic natural language summaries for charts, figures, and architecture diagrams using Gemini Vision."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        settings = get_settings()
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model = model or settings.GEMINI_MODEL
        self.prompt = DEFAULT_CHART_PROMPT
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

    def describe_figure(
        self,
        image_bytes: bytes,
        mime_type: str = "image/png",
        prompt: Optional[str] = None,
    ) -> str:
        """
        Generate a forensic textual summary for an image figure.
        Returns a fallback summary if offline or if no API key is provided.
        """
        if not self.api_key:
            return (
                "### 1. VISUAL IDENTITY & TITLE\n"
                "- Primary Title: [Offline Mock Figure]\n"
                "- Media Classification: Diagram\n\n"
                "### 2. LABELED ENTITY INVENTORY\n"
                "- Named Nodes: [Component A, Component B]\n\n"
                "### 5. FORENSIC SYNTHESIS\n"
                "- Primary Takeaway: Offline test figure indexed without live Gemini API key."
            )

        effective_prompt = prompt if prompt is not None else self.prompt
        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        url = f"{self.base_url}/models/{self.model}:generateContent"
        api_key = (self.api_key or "").strip()
        headers = {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": effective_prompt},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": b64_image,
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 1024,
            },
        }

        try:
            with httpx.Client(timeout=45.0) as client:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()

            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "").strip()

            return "[VLM: No content returned from Gemini Vision API]"

        except Exception as e:
            logger.error(f"[VLMVisionCaptioner] Failed to describe figure: {e}")
            return f"[VLM Extraction Error: {str(e)}]"
