"""
Vision Agents — Multi-Provider Ensemble
========================================
ארכיטקטורה: כל סוכן Vision שולח לכל ה-providers הזמינים במקביל,
משווה תוצאות, ומדווח על קונפליקטים ל-Red Team.

סטטוס providers:
  Claude (Anthropic)  -- פעיל
  Gemini (Google)     -- HOLD (מוכן, מחכה לאישור הפעלה)
  OpenAI GPT-4o-mini  -- HOLD (מוכן, מחכה לאישור הפעלה)

כדי להפעיל provider: שנה את PROVIDER_FLAGS בתחתית הקובץ.
"""

import os
import base64
import json
import asyncio
import httpx


# ============================================================
# PROVIDER FLAGS
# שנה ל-True כדי להפעיל provider.
# Claude תמיד פעיל. שאר ה-providers על HOLD עד אישור.
# ============================================================
PROVIDER_FLAGS = {
    "claude":  True,   # פעיל תמיד
    "gemini":  False,  # HOLD -- שנה ל-True כשמאשרים
    "openai":  False,  # HOLD -- שנה ל-True כשמאשרים
}

# משקל כל provider בציון הסופי
PROVIDER_WEIGHTS = {
    "claude":  0.50,
    "gemini":  0.30,
    "openai":  0.20,
}

# הפרש ציון מינימלי לזיהוי קונפליקט בין providers
CONFLICT_THRESHOLD = 0.25


# ============================================================
# קריאות ל-APIs
# ============================================================

async def _call_claude(file_bytes: bytes, system_prompt: str, user_prompt: str):
    """קריאה ל-Claude Vision (Anthropic)."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None

    b64 = base64.b64encode(file_bytes).decode()
    media_type = "image/jpeg"
    if file_bytes[:8].startswith(b'\x89PNG'):
        media_type = "image/png"
    elif file_bytes[:4] == b'RIFF':
        media_type = "image/webp"

    try:
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
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": b64,
                            }},
                            {"type": "text", "text": user_prompt}
                        ]
                    }]
                }
            )
            data = resp.json()
            if "content" in data and len(data["content"]) > 0:
                return {"provider": "claude", "text": data["content"][0].get("text", "")}
    except Exception as e:
        print(f"[Claude] error: {e}")
    return None


async def _call_gemini(file_bytes: bytes, system_prompt: str, user_prompt: str):
    """
    HOLD -- קריאה ל-Gemini 2.0 Flash Vision (Google).
    מוכן לחלוטין. יופעל כש-PROVIDER_FLAGS['gemini'] = True.
    """
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return None

    b64 = base64.b64encode(file_bytes).decode()
    mime_type = "image/jpeg"
    if file_bytes[:8].startswith(b'\x89PNG'):
        mime_type = "image/png"
    elif file_bytes[:4] == b'RIFF':
        mime_type = "image/webp"

    full_prompt = f"{system_prompt}\n\n{user_prompt}"

    try:
        async with httpx.AsyncClient(timeout=90) as client:
            resp = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}",
                headers={"content-type": "application/json"},
                json={
                    "contents": [{
                        "parts": [
                            {"inline_data": {"mime_type": mime_type, "data": b64}},
                            {"text": full_prompt}
                        ]
                    }],
                    "generationConfig": {"maxOutputTokens": 2000, "temperature": 0.1}
                }
            )
            data = resp.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return {"provider": "gemini", "text": text}
    except Exception as e:
        print(f"[Gemini] error: {e}")
    return None


async def _call_openai(file_bytes: bytes, system_prompt: str, user_prompt: str):
    """
    HOLD -- קריאה ל-GPT-4o-mini Vision (OpenAI).
    מוכן לחלוטין. יופעל כש-PROVIDER_FLAGS['openai'] = True.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return None

    b64 = base64.b64encode(file_bytes).decode()

    try:
        async with httpx.AsyncClient(timeout=90) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "gpt-4o-mini",
                    "max_tokens": 2000,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": [
                            {"type": "text", "text": user_prompt},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}",
                                "detail": "high"
                            }}
                        ]}
                    ]
                }
            )
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            return {"provider": "openai", "text": text}
    except Exception as e:
        print(f"[OpenAI] error: {e}")
    return None


# ============================================================
# מנוע ה-Ensemble
# ============================================================

def _parse_json_response(text: str):
    """מנתח תשובת JSON מכל provider."""
    try:
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(text)
    except Exception:
        return None


async def _call_ensemble(file_bytes: bytes, system_prompt: str, user_prompt: str) -> dict:
    """
    שולח לכל ה-providers הפעילים (לפי PROVIDER_FLAGS) במקביל.
    מחזיר dict עם תשובות גולמיות ורשימת providers שהגיבו.
    """
    tasks = {}
    if PROVIDER_FLAGS.get("claude"):
        tasks["claude"] = _call_claude(file_bytes, system_prompt, user_prompt)
    if PROVIDER_FLAGS.get("gemini"):
        tasks["gemini"] = _call_gemini(file_bytes, system_prompt, user_prompt)
    if PROVIDER_FLAGS.get("openai"):
        tasks["openai"] = _call_openai(file_bytes, system_prompt, user_prompt)

    if not tasks:
        return {"responses": [], "providers_used": []}

    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    responses = []
    for name, result in zip(tasks.keys(), results):
        if isinstance(result, Exception):
            print(f"[Ensemble] {name} raised: {result}")
        elif result is not None:
            responses.append(result)

    return {
        "responses": responses,
        "providers_used": [r["provider"] for r in responses]
    }


def _merge_parsed_results(parsed_list: list) -> dict:
    """
    ממזג תוצאות מ-providers מרובים.
    - ציון: ממוצע משוקלל לפי PROVIDER_WEIGHTS
    - אנומליות: איחוד עם תיוג מקור
    - קונפליקט: מחושב אם הפרש ציון >= CONFLICT_THRESHOLD
    """
    if not parsed_list:
        return {}
    if len(parsed_list) == 1:
        return parsed_list[0]["parsed"]

    total_weight = 0.0
    weighted_score = 0.0
    all_anomalies = []
    summaries = []
    scores = {}

    for item in parsed_list:
        provider = item["provider"]
        parsed = item["parsed"]
        weight = PROVIDER_WEIGHTS.get(provider, 0.33)

        score = parsed.get("confidence_score") or parsed.get("confidence") or 0.7
        scores[provider] = score
        weighted_score += score * weight
        total_weight += weight

        anomalies = parsed.get("anomalies") or parsed.get("indicators") or []
        for anomaly in anomalies:
            anomaly["_source_provider"] = provider
        all_anomalies.extend(anomalies)

        summary = parsed.get("summary", "")
        if summary:
            summaries.append(f"[{provider}] {summary}")

    final_score = round(weighted_score / total_weight, 3) if total_weight > 0 else 0.5
    conflict = (max(scores.values()) - min(scores.values())) >= CONFLICT_THRESHOLD if len(scores) >= 2 else False

    base = parsed_list[0]["parsed"].copy()
    base["confidence_score"] = final_score
    base["confidence"] = final_score
    base["anomalies"] = all_anomalies
    base["indicators"] = all_anomalies
    base["summary"] = " | ".join(summaries) if summaries else base.get("summary", "")
    base["_ensemble"] = {
        "providers": list(scores.keys()),
        "scores": scores,
        "consensus_conflict": conflict,
    }
    return base


# ============================================================
# סוכן פיזיקלי
# ============================================================
class PhysicalAgent:
    AGENT_TYPE = "physical"

    SYSTEM_PROMPT = """You are a forensic physics expert analyzing images for authenticity.
Analyze the image and look for:
1. Shadow direction inconsistencies between objects
2. Lighting issues -- conflicting light sources
3. Perspective errors -- mismatched vanishing points
4. Inconsistent reflections
5. Unnatural proportions

RESPOND ONLY WITH JSON in this exact format:
{"anomalies": [{"type": "shadows/lighting/perspective/reflections/proportions", "description": "detailed description in Hebrew", "severity": "high/medium/low", "location": {"x": 0-100, "y": 0-100}}], "confidence_score": 0.0-1.0, "summary": "short summary in Hebrew"}

If the image appears authentic with no issues, return empty anomalies array and high confidence_score."""

    USER_PROMPT = "Analyze this image for physical inconsistencies -- shadows, lighting, perspective, reflections, and proportions. Be thorough but avoid false positives."

    async def analyze(self, file_bytes: bytes, filename: str) -> dict:
        ensemble = await _call_ensemble(file_bytes, self.SYSTEM_PROMPT, self.USER_PROMPT)

        parsed_list = []
        for r in ensemble["responses"]:
            parsed = _parse_json_response(r["text"])
            if parsed:
                parsed_list.append({"provider": r["provider"], "parsed": parsed})

        if parsed_list:
            merged = _merge_parsed_results(parsed_list)
            result = {
                "agent_type": self.AGENT_TYPE,
                "confidence_score": merged.get("confidence_score", 0.7),
                "findings": {
                    "summary": merged.get("summary", ""),
                    "source": "+".join(ensemble["providers_used"]),
                    "ensemble": merged.get("_ensemble", {})
                },
                "anomalies": {"items": merged.get("anomalies", [])}
            }
            if merged.get("_ensemble", {}).get("consensus_conflict"):
                result["consensus_conflict"] = True
            return result

        # Mock fallback
        return {
            "agent_type": self.AGENT_TYPE,
            "confidence_score": 0.68,
            "findings": {"summary": "ניתוח מבוסס mock -- חבר API key לניתוח אמיתי", "source": "mock"},
            "anomalies": {"items": [
                {"type": "צללים", "description": "כיוון הצל של האובייקט המרכזי סותר את כיוון הצל ברקע.", "severity": "high", "location": {"x": 50, "y": 60}},
                {"type": "פרספקטיבה", "description": "נקודות ההיעלמות אינן תואמות את הפרספקטיבה של האובייקטים.", "severity": "medium", "location": {"x": 25, "y": 75}}
            ]}
        }


# ============================================================
# סוכן הקשרי
# ============================================================
class ContextualAgent:
    AGENT_TYPE = "contextual"

    SYSTEM_PROMPT = """You are a historical and contextual forensic expert. Analyze the image for:
1. Elements that don't match the apparent time period (uniforms, weapons, technology, vehicles)
2. Architectural styles that don't match the location or era
3. Vegetation inconsistent with the geographic region
4. Anachronistic typography, signs, or symbols
5. Clothing styles, hairstyles, or accessories that don't fit

RESPOND ONLY WITH JSON in this exact format:
{"anomalies": [{"type": "period/uniforms/technology/architecture/vegetation", "description": "detailed description in Hebrew", "severity": "high/medium/low", "location": {"x": 0-100, "y": 0-100}}], "confidence_score": 0.0-1.0, "summary": "short summary in Hebrew"}

If nothing appears anachronistic, return empty anomalies and high confidence_score."""

    USER_PROMPT = "Analyze this image for historical and contextual inconsistencies. Identify any element that doesn't belong to the apparent time period, location, or cultural context."

    async def analyze(self, file_bytes: bytes, filename: str) -> dict:
        ensemble = await _call_ensemble(file_bytes, self.SYSTEM_PROMPT, self.USER_PROMPT)

        parsed_list = []
        for r in ensemble["responses"]:
            parsed = _parse_json_response(r["text"])
            if parsed:
                parsed_list.append({"provider": r["provider"], "parsed": parsed})

        if parsed_list:
            merged = _merge_parsed_results(parsed_list)
            result = {
                "agent_type": self.AGENT_TYPE,
                "confidence_score": merged.get("confidence_score", 0.7),
                "findings": {
                    "summary": merged.get("summary", ""),
                    "source": "+".join(ensemble["providers_used"]),
                    "ensemble": merged.get("_ensemble", {})
                },
                "anomalies": {"items": merged.get("anomalies", [])}
            }
            if merged.get("_ensemble", {}).get("consensus_conflict"):
                result["consensus_conflict"] = True
            return result

        return {
            "agent_type": self.AGENT_TYPE,
            "confidence_score": 0.65,
            "findings": {"summary": "ניתוח מבוסס mock -- חבר API key לניתוח אמיתי", "source": "mock"},
            "anomalies": {"items": [
                {"type": "תקופה", "description": "אלמנטים בתמונה עשויים שלא להתאים לתקופה המצוינת.", "severity": "medium", "location": {"x": 60, "y": 40}}
            ]}
        }


# ============================================================
# סוכן זיהוי מנוע ייצור AI
# ============================================================
class AIGenerationAgent:
    AGENT_TYPE = "ai_generation"

    SYSTEM_PROMPT = """You are an expert in detecting AI-generated images. Analyze the image and determine:
1. Is this image AI-generated? (DALL-E, Midjourney, Stable Diffusion, Firefly, Sora, etc.)
2. If yes -- which tool/model most likely created it?
3. Key indicators: unnatural hands/fingers, distorted text, repetitive textures, asymmetric eyes, unnatural skin texture, impossible geometry, blurred backgrounds

RESPOND ONLY WITH JSON in this exact format:
{"is_ai_generated": true/false, "likely_tool": "tool name or unknown", "confidence": 0.0-1.0, "indicators": [{"type": "indicator name", "description": "description in Hebrew", "severity": "high/medium/low", "location": {"x": 0-100, "y": 0-100}}], "summary": "summary in Hebrew"}"""

    USER_PROMPT = "Determine if this image was generated by AI. If so, identify the likely tool and all telltale signs."

    async def analyze(self, file_bytes: bytes, filename: str) -> dict:
        ensemble = await _call_ensemble(file_bytes, self.SYSTEM_PROMPT, self.USER_PROMPT)

        parsed_list = []
        for r in ensemble["responses"]:
            parsed = _parse_json_response(r["text"])
            if parsed:
                parsed_list.append({"provider": r["provider"], "parsed": parsed})

        if parsed_list:
            merged = _merge_parsed_results(parsed_list)
            anomalies = merged.get("indicators") or merged.get("anomalies") or []
            if merged.get("is_ai_generated"):
                anomalies.insert(0, {
                    "type": "AI Generated",
                    "description": f"התמונה זוהתה כנוצרת על ידי AI. כלי משוער: {merged.get('likely_tool', 'לא ידוע')}.",
                    "severity": "high",
                    "location": {"x": 50, "y": 50}
                })
            result = {
                "agent_type": self.AGENT_TYPE,
                "confidence_score": merged.get("confidence", 0.7),
                "findings": {
                    "is_ai_generated": merged.get("is_ai_generated", False),
                    "likely_tool": merged.get("likely_tool", "unknown"),
                    "summary": merged.get("summary", ""),
                    "source": "+".join(ensemble["providers_used"]),
                    "ensemble": merged.get("_ensemble", {})
                },
                "anomalies": {"items": anomalies}
            }
            if merged.get("_ensemble", {}).get("consensus_conflict"):
                result["consensus_conflict"] = True
            return result

        return {
            "agent_type": self.AGENT_TYPE,
            "confidence_score": 0.70,
            "findings": {"is_ai_generated": False, "likely_tool": "unknown", "summary": "ניתוח מבוסס mock -- חבר API key לניתוח אמיתי", "source": "mock"},
            "anomalies": {"items": [
                {"type": "בדיקת AI", "description": "לא ניתן לקבוע בוודאות אם התמונה נוצרה על ידי AI ללא חיבור ל-Vision API.", "severity": "low", "location": {"x": 50, "y": 50}}
            ]}
        }
