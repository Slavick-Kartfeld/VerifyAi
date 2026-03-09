"""
Audio Deepfake Detection Agent.
שני מסלולים:
1. ניתוח ספקטרלי לוקלי (בלי API) — רעש רקע, חתכי תדר, עקביות
2. Claude API לניתוח דיבור (כשיש API key)
"""
import io
import os
import struct
import base64
import json
import numpy as np
import httpx


class AudioDeepfakeAgent:
    """זיהוי אודיו סינתטי / deepfake."""

    AGENT_TYPE = "audio_deepfake"

    SYSTEM_PROMPT = """You are an expert in audio forensics and deepfake detection.
You are given a description of audio spectral characteristics. Analyze for:
1. Signs of text-to-speech (TTS) synthesis — unnaturally perfect pitch, robotic transitions
2. Voice cloning artifacts — spectral gaps, unnatural formants
3. Audio splicing — abrupt frequency changes, mismatched room acoustics
4. Noise floor anomalies — AI audio often lacks natural ambient noise

RESPOND ONLY WITH JSON:
{"is_synthetic": true/false, "confidence": 0.0-1.0, "likely_method": "TTS/voice_clone/splice/natural",
"indicators": [{"type": "indicator", "description": "in English", "severity": "high/medium/low"}],
"summary": "brief summary in English"}"""

    async def analyze(self, file_bytes: bytes, filename: str) -> dict:
        anomalies = []
        findings = {}

        # 1. Local spectral analysis (works without API)
        spectral = self._spectral_analysis(file_bytes, filename)
        anomalies.extend(spectral.get("anomalies", []))
        findings.update(spectral.get("findings", {}))

        # 2. Try Claude API for deeper analysis
        api_result = await self._call_claude_audio(spectral)
        if api_result:
            anomalies.extend(api_result.get("anomalies", []))
            findings["ai_analysis"] = api_result.get("summary", "")
            findings["source"] = "claude_vision"
            confidence = api_result.get("confidence", 0.5)
        else:
            findings["source"] = "local_analysis"
            confidence = spectral.get("confidence", 0.6)

        return {
            "agent_type": self.AGENT_TYPE,
            "confidence_score": round(confidence, 2),
            "findings": findings,
            "anomalies": {"items": anomalies},
        }

    def _spectral_analysis(self, file_bytes: bytes, filename: str) -> dict:
        """ניתוח ספקטרלי בסיסי של אודיו."""
        anomalies = []
        findings = {}

        try:
            # Try to parse WAV
            samples, sample_rate = self._parse_wav(file_bytes)
            if samples is None:
                return {
                    "anomalies": [{
                        "type": "Format",
                        "description": "Unable to parse audio file. Only WAV format supported for local analysis.",
                        "severity": "low",
                        "location": {"x": 50, "y": 50},
                    }],
                    "findings": {"format": "unsupported"},
                    "confidence": 0.5,
                }

            findings["sample_rate"] = sample_rate
            findings["duration_sec"] = round(len(samples) / sample_rate, 2)
            findings["samples"] = len(samples)

            # Normalize
            samples = samples.astype(np.float64)
            if np.max(np.abs(samples)) > 0:
                samples = samples / np.max(np.abs(samples))

            # 1. Silence ratio — synthetic audio often has perfect silence
            silence_ratio = np.mean(np.abs(samples) < 0.01)
            findings["silence_ratio"] = round(silence_ratio, 3)
            if silence_ratio > 0.7:
                anomalies.append({
                    "type": "Excessive Silence",
                    "description": f"Audio is {silence_ratio:.0%} silence. Natural recordings typically have ambient noise.",
                    "severity": "medium",
                    "location": {"x": 50, "y": 30},
                })

            # 2. Frequency analysis via FFT
            if len(samples) > 1024:
                fft = np.fft.rfft(samples[:min(len(samples), sample_rate * 5)])  # First 5 sec
                freqs = np.fft.rfftfreq(len(samples[:min(len(samples), sample_rate * 5)]), 1.0 / sample_rate)
                magnitude = np.abs(fft)

                # Check for sharp frequency cutoff (common in TTS)
                if len(magnitude) > 100:
                    # Energy above 8kHz vs total
                    idx_8k = np.searchsorted(freqs, 8000)
                    if idx_8k < len(magnitude):
                        high_energy = np.mean(magnitude[idx_8k:])
                        total_energy = np.mean(magnitude)
                        hf_ratio = high_energy / (total_energy + 1e-10)
                        findings["hf_energy_ratio"] = round(hf_ratio, 4)

                        if hf_ratio < 0.02 and sample_rate >= 16000:
                            anomalies.append({
                                "type": "Frequency Cutoff",
                                "description": f"Sharp high-frequency cutoff detected. Energy above 8kHz is only "
                                             f"{hf_ratio:.2%} of total. TTS systems often produce limited bandwidth audio.",
                                "severity": "high",
                                "location": {"x": 80, "y": 50},
                            })

                    # Check for unnatural spectral peaks (TTS artifacts)
                    # Divide spectrum into bands and check for outlier peaks
                    band_size = len(magnitude) // 20
                    if band_size > 10:
                        band_energies = [np.mean(magnitude[i*band_size:(i+1)*band_size])
                                       for i in range(20)]
                        band_std = np.std(band_energies)
                        band_mean = np.mean(band_energies)
                        if band_mean > 0:
                            cv = band_std / band_mean  # coefficient of variation
                            findings["spectral_cv"] = round(cv, 3)
                            if cv > 2.5:
                                anomalies.append({
                                    "type": "Spectral Anomaly",
                                    "description": f"Unusual spectral energy distribution (CV={cv:.2f}). "
                                                 f"Natural speech has smoother spectral envelope.",
                                    "severity": "medium",
                                    "location": {"x": 50, "y": 50},
                                })

            # 3. Zero-crossing rate analysis (synthetic audio has more regular patterns)
            if len(samples) > 4096:
                chunk = 2048
                zcrs = []
                for start in range(0, len(samples) - chunk, chunk):
                    segment = samples[start:start + chunk]
                    zcr = np.sum(np.abs(np.diff(np.sign(segment))) > 0) / chunk
                    zcrs.append(zcr)

                if len(zcrs) > 4:
                    zcr_std = np.std(zcrs)
                    zcr_mean = np.mean(zcrs)
                    findings["zcr_mean"] = round(zcr_mean, 4)
                    findings["zcr_std"] = round(zcr_std, 4)

                    if zcr_std < 0.005 and zcr_mean > 0.05:
                        anomalies.append({
                            "type": "Uniform Zero-Crossing",
                            "description": f"Zero-crossing rate is unusually uniform (std={zcr_std:.4f}). "
                                         f"Natural speech has variable rhythm. This may indicate synthesis.",
                            "severity": "medium",
                            "location": {"x": 30, "y": 70},
                        })

            # Calculate confidence
            high_count = sum(1 for a in anomalies if a["severity"] == "high")
            confidence = max(0.2, 0.85 - high_count * 0.15 - len(anomalies) * 0.06)

            return {"anomalies": anomalies, "findings": findings, "confidence": confidence}

        except Exception as e:
            return {"anomalies": [], "findings": {"error": str(e)}, "confidence": 0.5}

    def _parse_wav(self, file_bytes: bytes):
        """פרסור בסיסי של WAV."""
        try:
            if len(file_bytes) < 44:
                return None, None
            if file_bytes[:4] != b'RIFF' or file_bytes[8:12] != b'WAVE':
                return None, None

            # Parse header
            channels = struct.unpack_from('<H', file_bytes, 22)[0]
            sample_rate = struct.unpack_from('<I', file_bytes, 24)[0]
            bits_per_sample = struct.unpack_from('<H', file_bytes, 34)[0]

            # Find data chunk
            pos = 12
            while pos < len(file_bytes) - 8:
                chunk_id = file_bytes[pos:pos + 4]
                chunk_size = struct.unpack_from('<I', file_bytes, pos + 4)[0]
                if chunk_id == b'data':
                    data_start = pos + 8
                    data_end = min(data_start + chunk_size, len(file_bytes))
                    raw = file_bytes[data_start:data_end]

                    if bits_per_sample == 16:
                        samples = np.frombuffer(raw, dtype=np.int16)
                    elif bits_per_sample == 8:
                        samples = np.frombuffer(raw, dtype=np.uint8).astype(np.int16) - 128
                    else:
                        return None, None

                    # Convert to mono if stereo
                    if channels == 2 and len(samples) > 1:
                        samples = samples[::2]

                    return samples, sample_rate
                pos += 8 + chunk_size
                if chunk_size % 2 == 1:
                    pos += 1  # padding byte

            return None, None
        except Exception:
            return None, None

    async def _call_claude_audio(self, spectral_data: dict) -> dict | None:
        """שליחת תיאור ספקטרלי ל-Claude לניתוח מעמיק."""
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            return None

        findings = spectral_data.get("findings", {})
        if not findings or findings.get("format") == "unsupported":
            return None

        # Build description for Claude
        desc = f"""Audio file analysis results:
- Sample rate: {findings.get('sample_rate', 'N/A')} Hz
- Duration: {findings.get('duration_sec', 'N/A')} seconds
- Silence ratio: {findings.get('silence_ratio', 'N/A')}
- HF energy ratio (>8kHz): {findings.get('hf_energy_ratio', 'N/A')}
- Spectral CV: {findings.get('spectral_cv', 'N/A')}
- Zero-crossing rate mean: {findings.get('zcr_mean', 'N/A')}
- Zero-crossing rate std: {findings.get('zcr_std', 'N/A')}
- Local anomalies found: {len(spectral_data.get('anomalies', []))}"""

        try:
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
                        "max_tokens": 1500,
                        "system": self.SYSTEM_PROMPT,
                        "messages": [{"role": "user", "content": desc}],
                    }
                )
                data = resp.json()
                if "content" in data and len(data["content"]) > 0:
                    text = data["content"][0].get("text", "")
                    text = text.strip()
                    if text.startswith("```"):
                        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
                    parsed = json.loads(text)

                    result_anomalies = []
                    for ind in parsed.get("indicators", []):
                        result_anomalies.append({
                            "type": ind.get("type", "Audio"),
                            "description": ind.get("description", ""),
                            "severity": ind.get("severity", "medium"),
                            "location": {"x": 50, "y": 50},
                        })

                    if parsed.get("is_synthetic"):
                        result_anomalies.insert(0, {
                            "type": "Synthetic Audio",
                            "description": f"Audio identified as synthetic. Likely method: {parsed.get('likely_method', 'unknown')}.",
                            "severity": "high",
                            "location": {"x": 50, "y": 50},
                        })

                    return {
                        "anomalies": result_anomalies,
                        "confidence": parsed.get("confidence", 0.5),
                        "summary": parsed.get("summary", ""),
                    }
        except Exception:
            pass

        return None
