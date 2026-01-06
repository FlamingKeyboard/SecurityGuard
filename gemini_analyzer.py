"""Gemini vision analysis for security assessment using google-genai SDK with structured outputs."""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

import config

_LOGGER = logging.getLogger(__name__)


# Pydantic model for structured output - Gemini will be forced to match this schema
class WeaponDetection(BaseModel):
    """Weapon detection result."""
    detected: bool = Field(description="Whether a weapon is visible")
    confidence: float = Field(description="Confidence level 0.0-1.0", ge=0.0, le=1.0)
    description: str = Field(default="", description="Brief description if weapon detected")


class SecurityAnalysisResponse(BaseModel):
    """Structured response schema for security analysis."""
    risk_tier: Literal["low", "medium", "high"] = Field(
        description="Risk level: low=normal activity, medium=unusual but not threatening, high=clearly concerning"
    )
    person_detected: bool = Field(description="Whether a person is visible in the image")
    person_count: int = Field(default=0, description="Number of people visible", ge=0)
    activity_observed: list[str] = Field(
        default_factory=list,
        description="List of observed activities (e.g., 'walking', 'carrying package', 'looking at door')"
    )
    potential_concerns: list[str] = Field(
        default_factory=list,
        description="List of any concerning observations"
    )
    context_clues: list[str] = Field(
        default_factory=list,
        description="Context clues like 'delivery uniform', 'package', 'tools', 'vehicle'"
    )
    weapon_visible: WeaponDetection = Field(
        default_factory=lambda: WeaponDetection(detected=False, confidence=0.0, description="")
    )
    time_of_day_apparent: Literal["day", "night", "unclear"] = Field(
        default="unclear",
        description="Apparent time of day based on lighting"
    )
    recommended_action: Literal["ignore", "log", "notify", "urgent_alert"] = Field(
        description="Recommended action based on analysis"
    )
    summary: str = Field(description="One sentence factual summary of what is visible")


# Analysis prompt - simpler since schema is enforced
SECURITY_ANALYSIS_PROMPT = """Analyze this security camera image. Focus on OBSERVABLE FACTS only.

Risk tier guidelines:
- low: Normal activity (delivery person, neighbor, mail carrier, resident, empty scene)
- medium: Unusual but not clearly threatening (unfamiliar person lingering, someone looking at windows)
- high: Clearly concerning (attempted entry, visible weapon, aggressive behavior, property damage)

Be objective and evidence-based. If uncertain, note it in your response."""


@dataclass
class SecurityAnalysis:
    """Result of security analysis (for internal use)."""
    risk_tier: Literal["low", "medium", "high"]
    person_detected: bool
    person_count: int
    activity_observed: list[str]
    potential_concerns: list[str]
    context_clues: list[str]
    weapon_visible: dict
    time_of_day_apparent: str
    recommended_action: str
    summary: str
    raw_response: str


def get_gemini_api_key() -> str | None:
    """Get Gemini API key from env or stored credentials."""
    import os
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        try:
            from vivint_client import get_stored_credential
            key = get_stored_credential("gemini_api_key")
        except Exception:
            pass
    return key


def validate_api_key(key: str) -> bool:
    """Validate that the API key has the correct format."""
    if not key:
        return False
    return key.startswith("AIza")


def _create_client():
    """Create and return a Gemini client."""
    api_key = get_gemini_api_key()
    if not api_key:
        _LOGGER.error("No Gemini API key found. Run setup_credentials.py to configure.")
        return None

    if not validate_api_key(api_key):
        _LOGGER.error(
            "Invalid Gemini API key format. Keys should start with 'AIza'. "
            "Get a valid key from https://aistudio.google.com/apikey"
        )
        return None

    try:
        from google import genai
        return genai.Client(api_key=api_key)
    except Exception as e:
        _LOGGER.error("Failed to create Gemini client: %s", e)
        return None


async def analyze_frame(frame_path: Path) -> SecurityAnalysis | None:
    """
    Analyze a single frame using Gemini vision with structured output.

    Args:
        frame_path: Path to the image file

    Returns:
        SecurityAnalysis object or None if analysis failed
    """
    client = _create_client()
    if not client:
        return None

    try:
        from google.genai import types

        # Read image
        image_bytes = frame_path.read_bytes()

        # Make request with structured output
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=config.GEMINI_MODEL,
            contents=[
                SECURITY_ANALYSIS_PROMPT,
                types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg'),
            ],
            config={
                "response_mime_type": "application/json",
                "response_schema": SecurityAnalysisResponse,
            },
        )

        raw_text = response.text

        # Parse the structured response
        import json
        data = json.loads(raw_text)

        # Handle weapon_visible which may be a dict or need conversion
        weapon_data = data.get("weapon_visible", {})
        if isinstance(weapon_data, dict):
            weapon_dict = weapon_data
        else:
            weapon_dict = {"detected": False, "confidence": 0.0, "description": ""}

        return SecurityAnalysis(
            risk_tier=data.get("risk_tier", "low"),
            person_detected=data.get("person_detected", False),
            person_count=data.get("person_count", 0),
            activity_observed=data.get("activity_observed", []),
            potential_concerns=data.get("potential_concerns", []),
            context_clues=data.get("context_clues", []),
            weapon_visible=weapon_dict,
            time_of_day_apparent=data.get("time_of_day_apparent", "unclear"),
            recommended_action=data.get("recommended_action", "log"),
            summary=data.get("summary", ""),
            raw_response=raw_text,
        )

    except json.JSONDecodeError as e:
        _LOGGER.error("Failed to parse Gemini response as JSON: %s", e)
        _LOGGER.debug("Raw response: %s", raw_text if 'raw_text' in locals() else "N/A")
        return None
    except Exception as e:
        _LOGGER.error("Error analyzing frame with Gemini: %s", e)
        return None


async def analyze_multiple_frames(frame_paths: list[Path]) -> SecurityAnalysis | None:
    """
    Analyze multiple frames and return the highest-risk assessment.
    """
    if not frame_paths:
        return None

    analyses = []
    for path in frame_paths:
        analysis = await analyze_frame(path)
        if analysis:
            analyses.append(analysis)

    if not analyses:
        return None

    # Return the highest-risk analysis
    risk_order = {"high": 3, "medium": 2, "low": 1}
    return max(analyses, key=lambda a: risk_order.get(a.risk_tier, 0))


async def test_analysis(image_path: str):
    """Test analysis on a single image."""
    logging.basicConfig(level=logging.INFO)

    path = Path(image_path)
    if not path.exists():
        print(f"Image not found: {path}")
        return

    print(f"Analyzing: {path}")
    analysis = await analyze_frame(path)

    if analysis:
        print("\n=== Security Analysis ===")
        print(f"Risk Tier: {analysis.risk_tier.upper()}")
        print(f"Person Detected: {analysis.person_detected} (count: {analysis.person_count})")
        print(f"Activities: {', '.join(analysis.activity_observed) or 'None observed'}")
        print(f"Concerns: {', '.join(analysis.potential_concerns) or 'None'}")
        print(f"Context Clues: {', '.join(analysis.context_clues) or 'None'}")
        print(f"Weapon Visible: {analysis.weapon_visible}")
        print(f"Time of Day: {analysis.time_of_day_apparent}")
        print(f"Recommended Action: {analysis.recommended_action}")
        print(f"\nSummary: {analysis.summary}")
        print(f"\nRaw JSON:\n{analysis.raw_response}")
    else:
        print("Analysis failed")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        asyncio.run(test_analysis(sys.argv[1]))
    else:
        print("Usage: python gemini_analyzer.py <image_path>")
