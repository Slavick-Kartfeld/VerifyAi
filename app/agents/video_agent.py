"""
Video Forensic Agent — ניתוח וידאו פורנזי.
מחלץ פריימים ומנתח:
- עקביות טמפורלית (זיהוי קפיצות/חיתוכים חשודים)
- עקביות תאורה בין פריימים
- זיהוי שינויי רזולוציה/איכות פתאומיים
- Claude Vision לניתוח פריימים חשודים (עם API key)
"""
import io
import os
import base64
import json
import struct
import numpy as np
from PIL import Image
import httpx


class VideoForensicAgent:
    """ניתוח פורנזי של קבצי וידאו."""

    AGENT_TYPE = "video_forensic"

    async def analyze(self, file_bytes: bytes, filename: str) -> dict:
        anomalies = []
        findings = {}

        # Extract frames from video
        frames = self._extract_frames_basic(file_bytes, filename)
        findings["frames_extracted"] = len(frames)

        if len(frames) < 2:
            findings["source"] = "insufficient_frames"
            return {
                "agent_type": self.AGENT_TYPE,
                "confidence_score": 0.5,
                "findings": findings,
                "anomalies": {"items": [{
                    "type": "Frame Extraction",
                    "description": "Could not extract sufficient frames for analysis. "
                                 "Full video analysis requires ffmpeg (production deployment).",
                    "severity": "low",
                    "location": {"x": 50, "y": 50},
                }]},
            }

        # 1. Temporal consistency — brightness/contrast changes
        temporal = self._temporal_consistency(frames)
        anomalies.extend(temporal["anomalies"])
        findings["temporal_score"] = temporal.get("score", 0)

        # 2. Frame similarity — detect sudden jumps
        similarity = self._frame_similarity(frames)
        anomalies.extend(similarity["anomalies"])
        findings["similarity_score"] = similarity.get("score", 0)

        # 3. Resolution consistency
        res = self._resolution_consistency(frames)
        anomalies.extend(res["anomalies"])

        # 4. Claude Vision on suspicious frames (if available)
        if anomalies:
            api_result = await self._analyze_suspicious_frame(frames, anomalies)
            if api_result:
                anomalies.extend(api_result.get("anomalies", []))
                findings["ai_analysis"] = api_result.get("summary", "")
                findings["source"] = "claude_vision"
            else:
                findings["source"] = "local_analysis"
        else:
            findings["source"] = "local_analysis"

        # Confidence
        high_count = sum(1 for a in anomalies if a["severity"] == "high")
        confidence = max(0.15, 0.85 - high_count * 0.15 - len(anomalies) * 0.05)

        return {
            "agent_type": self.AGENT_TYPE,
            "confidence_score": round(confidence, 2),
            "findings": findings,
            "anomalies": {"items": anomalies},
        }

    def _extract_frames_basic(self, file_bytes: bytes, filename: str) -> list:
        """Extract frames from video — basic approach without ffmpeg.
        In production, use ffmpeg for proper decoding.
        For MVP, we analyze the raw bytes for JPEG markers in MP4/AVI."""
        frames = []

        # Look for JPEG frames embedded in video container
        # JPEG starts with FF D8 FF, ends with FF D9
        data = file_bytes
        pos = 0
        max_frames = 10  # Extract up to 10 frames

        while pos < len(data) - 4 and len(frames) < max_frames:
            # Find JPEG start
            idx = data.find(b'\xff\xd8\xff', pos)
            if idx == -1:
                break

            # Find JPEG end
            end_idx = data.find(b'\xff\xd9', idx + 3)
            if end_idx == -1:
                break

            end_idx += 2  # Include the marker
            jpeg_data = data[idx:end_idx]

            if len(jpeg_data) > 1000:  # Minimum viable JPEG
                try:
                    img = Image.open(io.BytesIO(jpeg_data))
                    if img.width >= 32 and img.height >= 32:
                        # Resize for analysis
                        img = img.convert("L")
                        ratio = min(256 / img.width, 256 / img.height, 1.0)
                        if ratio < 1.0:
                            img = img.resize((int(img.width * ratio), int(img.height * ratio)))
                        frames.append(np.array(img, dtype=np.float64))
                except Exception:
                    pass

            pos = end_idx

        return frames

    def _temporal_consistency(self, frames: list) -> dict:
        """Check brightness/contrast consistency across frames."""
        anomalies = []
        if len(frames) < 3:
            return {"anomalies": [], "score": 0.5}

        means = [f.mean() for f in frames]
        stds = [f.std() for f in frames]

        # Check for sudden brightness jumps
        mean_diffs = np.abs(np.diff(means))
        avg_diff = np.mean(mean_diffs) if len(mean_diffs) > 0 else 0

        score = 0
        for i, diff in enumerate(mean_diffs):
            if avg_diff > 0 and diff > avg_diff * 3 and diff > 15:
                pct = int((i + 1) / len(frames) * 100)
                anomalies.append({
                    "type": "Temporal Jump",
                    "description": f"Sudden brightness change between frames {i+1} and {i+2} "
                                 f"(delta: {diff:.1f}, average: {avg_diff:.1f}). "
                                 f"May indicate frame insertion or splice at ~{pct}% of video.",
                    "severity": "high" if diff > avg_diff * 5 else "medium",
                    "location": {"x": pct, "y": 50},
                })
                score += 0.2

        return {"anomalies": anomalies, "score": min(score, 1.0)}

    def _frame_similarity(self, frames: list) -> dict:
        """Check similarity between consecutive frames."""
        anomalies = []
        if len(frames) < 3:
            return {"anomalies": [], "score": 0}

        similarities = []
        for i in range(len(frames) - 1):
            f1, f2 = frames[i], frames[i + 1]
            # Resize to same dimensions if needed
            if f1.shape != f2.shape:
                min_h = min(f1.shape[0], f2.shape[0])
                min_w = min(f1.shape[1], f2.shape[1])
                f1 = f1[:min_h, :min_w]
                f2 = f2[:min_h, :min_w]

            # Normalized cross-correlation
            diff = np.mean(np.abs(f1 - f2))
            max_val = max(f1.max(), f2.max(), 1)
            similarity = 1.0 - (diff / max_val)
            similarities.append(similarity)

        if len(similarities) > 2:
            avg_sim = np.mean(similarities)
            # Look for outliers — frames that are very different from neighbors
            for i, sim in enumerate(similarities):
                if sim < avg_sim * 0.5 and sim < 0.7:
                    pct = int((i + 1) / len(frames) * 100)
                    anomalies.append({
                        "type": "Scene Discontinuity",
                        "description": f"Low similarity ({sim:.0%}) between frames {i+1} and {i+2} "
                                     f"vs average ({avg_sim:.0%}). Possible splice or deepfake transition.",
                        "severity": "high" if sim < 0.4 else "medium",
                        "location": {"x": pct, "y": 50},
                    })

                # Also check for identical frames (freeze/loop)
                if sim > 0.999 and avg_sim < 0.99:
                    pct = int((i + 1) / len(frames) * 100)
                    anomalies.append({
                        "type": "Frozen Frame",
                        "description": f"Frames {i+1} and {i+2} are nearly identical ({sim:.2%}). "
                                     f"May indicate frame duplication or loop insertion.",
                        "severity": "medium",
                        "location": {"x": pct, "y": 50},
                    })

        score = len(anomalies) * 0.15
        return {"anomalies": anomalies, "score": min(score, 1.0)}

    def _resolution_consistency(self, frames: list) -> dict:
        """Check if all frames have consistent dimensions."""
        anomalies = []
        if len(frames) < 2:
            return {"anomalies": []}

        shapes = [f.shape for f in frames]
        unique_shapes = set(shapes)

        if len(unique_shapes) > 1:
            anomalies.append({
                "type": "Resolution Mismatch",
                "description": f"Video contains frames with {len(unique_shapes)} different resolutions: "
                             f"{', '.join(f'{s[1]}x{s[0]}' for s in unique_shapes)}. "
                             f"Legitimate videos maintain consistent frame size.",
                "severity": "high",
                "location": {"x": 50, "y": 50},
            })

        return {"anomalies": anomalies}

    async def _analyze_suspicious_frame(self, frames: list, anomalies: list) -> dict | None:
        """Send suspicious frame to Claude Vision for analysis."""
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key or not frames:
            return None

        # Pick the frame most associated with anomalies
        frame = frames[min(len(frames) // 2, len(frames) - 1)]

        try:
            img = Image.fromarray(frame.astype(np.uint8))
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            b64 = base64.b64encode(buf.getvalue()).decode()

            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": "claude-sonnet-4-20250514",
                        "max_tokens": 1000,
                        "system": "You are a video forensics expert. Analyze this frame extracted from a video for signs of manipulation, deepfake, or compositing. RESPOND ONLY WITH JSON: {\"anomalies\": [{\"type\": \"...\", \"description\": \"...\", \"severity\": \"high/medium/low\"}], \"summary\": \"brief summary\"}",
                        "messages": [{
                            "role": "user",
                            "content": [
                                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                                {"type": "text", "text": "Analyze this video frame for signs of manipulation."},
                            ]
                        }]
                    }
                )
                data = resp.json()
                if "content" in data:
                    text = data["content"][0].get("text", "")
                    if text.startswith("```"):
                        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
                    parsed = json.loads(text.strip())
                    for a in parsed.get("anomalies", []):
                        a["location"] = a.get("location", {"x": 50, "y": 50})
                    return parsed
        except Exception:
            pass
        return None
