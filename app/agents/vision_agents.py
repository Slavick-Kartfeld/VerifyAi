"""
Vision API agents — Physical, Contextual, and AI Generation Detection.
Uses Claude Vision (Anthropic) by default.
Falls back to OpenAI Vision if key is available.
Returns mock data if no API key is set.
"""
import os
import base64
import json
import httpx


async def _call_claude_vision(file_bytes: bytes, system_prompt: str, user_prompt: str) -> str | None:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None

    b64 = base64.b64encode(file_bytes).decode()

    media_type = "image/jpeg"
    if file_bytes[:8].startswith(b'\x89PNG'):
        media_type = "image/png"
    elif file_bytes[:4] == b'RIFF':
        media_type = "image/webp"

    async with httpx.AsyncClient(timeout=90) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 2000,
                "system": system_prompt,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": b64,
                                }
                            },
                            {
                                "type": "text",
                                "text": user_prompt,
                            }
                        ]
                    }
                ]
            }
        )
        data = resp.json()
        if "content" in data and len(data["content"]) > 0:
            return data["content"][0].get("text", "")
        return None


async def _call_openai_vision(file_bytes: bytes, system_prompt: str, user_prompt: str) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return None

    b64 = base64.b64encode(file_bytes).decode()
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "gpt-4o",
                "max_tokens": 1500,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}", "detail": "high"
                        }}
                    ]}
                ]
            }
        )
        data = resp.json()
        return data["choices"][0]["message"]["content"]


async def _call_vision_api(file_bytes: bytes, system_prompt: str, user_prompt: str) -> str | None:
    result = await _call_claude_vision(file_bytes, system_prompt, user_prompt)
    if result:
        return result
    result = await _call_openai_vision(file_bytes, system_prompt, user_prompt)
    if result:
        return result
    return None


def _parse_json_response(text: str) -> dict | None:
    try:
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(text)
    except Exception:
        return None


# ========================================
# Physical Agent
# ========================================
class PhysicalAgent:
    AGENT_TYPE = "physical"

    SYSTEM_PROMPT = """You are a forensic physics expert analyzing images for authenticity.
Analyze the image and look for:
1. Shadow direction inconsistencies between objects
2. Lighting issues — conflicting light sources
3. Perspective errors — mismatched vanishing points
4. Inconsistent reflections
5. Unnatural proportions

RESPOND ONLY WITH JSON in this exact format:
{"anomalies": [{"type": "shadows/lighting/perspective/reflections/proportions", "description": "detailed description in English", "severity": "high/medium/low", "location": {"x": 0-100, "y": 0-100}}], "confidence_score": 0.0-1.0, "summary": "short summary in English"}

IMPORTANT: ALL text fields must be in English only. Do not use Hebrew or any other language.
If the image appears authentic with no issues, return empty anomalies array and high confidence_score."""

    USER_PROMPT = "Analyze this image for physical inconsistencies — shadows, lighting, perspective, reflections, and proportions. All descriptions must be in English only."

    async def analyze(self, file_bytes: bytes, filename: str) -> dict:
        response = await _call_vision_api(file_bytes, self.SYSTEM_PROMPT, self.USER_PROMPT)

        if response:
            parsed = _parse_json_response(response)
            if parsed:
                return {
                    "agent_type": self.AGENT_TYPE,
                    "confidence_score": parsed.get("confidence_score", 0.7),
                    "findings": {"summary": parsed.get("summary", ""), "source": "claude_vision"},
                    "anomalies": {"items": parsed.get("anomalies", [])}
                }

        return {
            "agent_type": self.AGENT_TYPE,
            "confidence_score": 0.68,
            "findings": {"summary": "Mock analysis — connect API key for real analysis", "source": "mock"},
            "anomalies": {"items": [
                {
                    "type": "Shadows",
                    "description": "Shadow direction of the central object contradicts background shadow direction. Significant gap suggests possible compositing.",
                    "severity": "high",
                    "location": {"x": 50, "y": 60}
                },
                {
                    "type": "Perspective",
                    "description": "Vanishing points of background lines do not match the perspective of foreground objects.",
                    "severity": "medium",
                    "location": {"x": 25, "y": 75}
                }
            ]}
        }


# ========================================
# Contextual Agent
# ========================================
class ContextualAgent:
    AGENT_TYPE = "contextual"

    SYSTEM_PROMPT = """You are a historical and contextual forensic expert. Analyze the image for:
1. Elements that don't match the apparent time period (uniforms, weapons, technology, vehicles)
2. Architectural styles that don't match the location or era
3. Vegetation inconsistent with the geographic region
4. Anachronistic typography, signs, or symbols
5. Clothing styles, hairstyles, or accessories that don't fit

RESPOND ONLY WITH JSON in this exact format:
{"anomalies": [{"type": "period/uniforms/technology/architecture/vegetation", "description": "detailed description in English", "severity": "high/medium/low", "location": {"x": 0-100, "y": 0-100}}], "confidence_score": 0.0-1.0, "summary": "short summary in English"}

IMPORTANT: ALL text fields must be in English only. Do not use Hebrew or any other language.
If nothing appears anachronistic, return empty anomalies and high confidence_score."""

    USER_PROMPT = "Analyze this image for historical and contextual inconsistencies. All descriptions must be in English only."

    async def analyze(self, file_bytes: bytes, filename: str) -> dict:
        response = await _call_vision_api(file_bytes, self.SYSTEM_PROMPT, self.USER_PROMPT)

        if response:
            parsed = _parse_json_response(response)
            if parsed:
                return {
                    "agent_type": self.AGENT_TYPE,
                    "confidence_score": parsed.get("confidence_score", 0.7),
                    "findings": {"summary": parsed.get("summary", ""), "source": "claude_vision"},
                    "anomalies": {"items": parsed.get("anomalies", [])}
                }

        return {
            "agent_type": self.AGENT_TYPE,
            "confidence_score": 0.65,
            "findings": {"summary": "Mock analysis — connect API key for real analysis", "source": "mock"},
            "anomalies": {"items": [
                {
                    "type": "Period",
                    "description": "Elements in the image may not match the stated time period. Deep analysis requires Vision API connection.",
                    "severity": "medium",
                    "location": {"x": 60, "y": 40}
                }
            ]}
        }


# ========================================
# AI Generation Detection Agent
# ========================================
class AIGenerationAgent:
    AGENT_TYPE = "ai_generation"

    SYSTEM_PROMPT = """You are an expert in detecting AI-generated images. Analyze the image and determine:
1. Is this image AI-generated? (DALL-E, Midjourney, Stable Diffusion, Firefly, etc.)
2. If yes — which tool/model most likely created it?
3. Key indicators: unnatural hands/fingers, distorted text, repetitive textures, asymmetric eyes, unnatural skin texture, impossible geometry, blurred backgrounds

RESPOND ONLY WITH JSON in this exact format:
{"is_ai_generated": true/false, "likely_tool": "tool name or unknown", "confidence": 0.0-1.0, "indicators": [{"type": "indicator name", "description": "description in English", "severity": "high/medium/low", "location": {"x": 0-100, "y": 0-100}}], "summary": "summary in English"}

IMPORTANT: ALL text fields must be in English only. Do not use Hebrew or any other language."""

    USER_PROMPT = "Determine if this image was generated by AI. All descriptions must be in English only."

    async def analyze(self, file_bytes: bytes, filename: str) -> dict:
        response = await _call_vision_api(file_bytes, self.SYSTEM_PROMPT, self.USER_PROMPT)

        if response:
            parsed = _parse_json_response(response)
            if parsed:
                anomalies = parsed.get("indicators", [])
                if parsed.get("is_ai_generated"):
                    anomalies.insert(0, {
                        "type": "AI Generated",
                        "description": f"Image identified as AI-generated. Likely tool: {parsed.get('likely_tool', 'unknown')}.",
                        "severity": "high",
                        "location": {"x": 50, "y": 50}
                    })
                return {
                    "agent_type": self.AGENT_TYPE,
                    "confidence_score": parsed.get("confidence", 0.7),
                    "findings": {
                        "is_ai_generated": parsed.get("is_ai_generated", False),
                        "likely_tool": parsed.get("likely_tool", "unknown"),
                        "summary": parsed.get("summary", ""),
                        "source": "claude_vision"
                    },
                    "anomalies": {"items": anomalies}
                }

        return {
            "agent_type": self.AGENT_TYPE,
            "confidence_score": 0.70,
            "findings": {
                "is_ai_generated": False,
                "likely_tool": "unknown",
                "summary": "Mock analysis — connect API key for real analysis",
                "source": "mock"
            },
            "anomalies": {"items": [
                {
                    "type": "AI Check",
                    "description": "Cannot determine AI generation without Vision API connection. Manual review recommended.",
                    "severity": "low",
                    "location": {"x": 50, "y": 50}
                }
            ]}
        }
