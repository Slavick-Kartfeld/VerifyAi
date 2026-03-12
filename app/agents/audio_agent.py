"""
Audio Deepfake Detection Agent.
פורמטים נתמכים: WAV (native), MP3/AAC/M4A/FLAC/OGG/OPUS/WMA/AIFF (via pydub+ffmpeg)
"""
import io
import os
import struct
import json
import numpy as np
import httpx


class AudioDeepfakeAgent:
    AGENT_TYPE = "audio_deepfake"

    SUPPORTED_FORMATS = {
        ".wav":  "wav",
        ".mp3":  "mp3",
        ".m4a":  "mp4",
        ".aac":  "aac",
        ".flac": "flac",
        ".ogg":  "ogg",
        ".opus": "opus",
        ".wma":  "asf",
        ".aiff": "aiff",
        ".aif":  "aiff",
    }

    SYSTEM_PROMPT = """You are an expert in audio forensics and deepfake detection.
You are given a description of audio spectral characteristics. Analyze for:
1. Signs of TTS synthesis — unnaturally perfect pitch, robotic transitions
2. Voice cloning artifacts — spectral gaps, unnatural formants
3. Audio splicing — abrupt frequency changes, mismatched room acoustics
4. Noise floor anomalies — AI audio often lacks natural ambient noise

RESPOND ONLY WITH JSON:
{"is_synthetic": true/false, "confidence": 0.0-1.0, "likely_method": "TTS/voice_clone/splice/natural",
"indicators": [{"type": "...", "description": "in English", "severity": "high/medium/low"}],
"summary": "brief summary in English"}"""

    async def analyze(self, file_bytes: bytes, filename: str) -> dict:
        anomalies = []
        findings = {}

        spectral = self._spectral_analysis(file_bytes, filename)
        anomalies.extend(spectral.get("anomalies", []))
        findings.update(spectral.get("findings", {}))

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

    def _parse_audio(self, file_bytes: bytes, filename: str):
        """Returns (samples, sample_rate, fmt). Tries WAV native first, then pydub."""
        ext = os.path.splitext(filename.lower())[1]

        # 1. Native WAV (no deps)
        samples, sr = self._parse_wav_native(file_bytes)
        if samples is not None:
            return samples, sr, "wav"

        # 2. pydub (needs ffmpeg for non-WAV)
        fmt = self.SUPPORTED_FORMATS.get(ext, ext.lstrip(".") or "wav")
        try:
            from pydub import AudioSegment
            seg = AudioSegment.from_file(io.BytesIO(file_bytes), format=fmt)
            seg = seg.set_channels(1)
            bps = seg.sample_width
            sr  = seg.frame_rate
            if bps == 2:
                samples = np.frombuffer(seg.raw_data, dtype=np.int16).astype(np.float64)
            elif bps == 4:
                samples = np.frombuffer(seg.raw_data, dtype=np.int32).astype(np.float64)
            else:
                samples = np.frombuffer(seg.raw_data, dtype=np.uint8).astype(np.float64) - 128
            return samples, sr, fmt
        except Exception:
            pass

        return None, None, ext or "unknown"

    def _parse_wav_native(self, file_bytes: bytes):
        try:
            if len(file_bytes) < 44:
                return None, None
            if file_bytes[:4] != b'RIFF' or file_bytes[8:12] != b'WAVE':
                return None, None
            channels        = struct.unpack_from('<H', file_bytes, 22)[0]
            sample_rate     = struct.unpack_from('<I', file_bytes, 24)[0]
            bits_per_sample = struct.unpack_from('<H', file_bytes, 34)[0]
            pos = 12
            while pos < len(file_bytes) - 8:
                chunk_id   = file_bytes[pos:pos + 4]
                chunk_size = struct.unpack_from('<I', file_bytes, pos + 4)[0]
                if chunk_id == b'data':
                    raw = file_bytes[pos + 8: min(pos + 8 + chunk_size, len(file_bytes))]
                    if bits_per_sample == 16:
                        samples = np.frombuffer(raw, dtype=np.int16)
                    elif bits_per_sample == 8:
                        samples = np.frombuffer(raw, dtype=np.uint8).astype(np.int16) - 128
                    else:
                        return None, None
                    if channels == 2 and len(samples) > 1:
                        samples = samples[::2]
                    return samples, sample_rate
                pos += 8 + chunk_size + (chunk_size % 2)
            return None, None
        except Exception:
            return None, None

    def _spectral_analysis(self, file_bytes: bytes, filename: str) -> dict:
        anomalies = []
        findings  = {}
        try:
            samples, sample_rate, fmt = self._parse_audio(file_bytes, filename)
            findings["detected_format"] = fmt

            if samples is None:
                ext = os.path.splitext(filename.lower())[1]
                supported = ", ".join(self.SUPPORTED_FORMATS.keys())
                return {
                    "anomalies": [{
                        "type": "Format Error",
                        "description": (
                            f"Unable to parse '{ext}' audio file. "
                            f"Supported: {supported}. "
                            "Ensure ffmpeg is installed on the server."
                        ),
                        "severity": "low",
                        "location": {"x": 50, "y": 50},
                    }],
                    "findings": {"format": "unsupported", "detected_ext": ext},
                    "confidence": 0.5,
                }

            findings["sample_rate"]  = sample_rate
            findings["duration_sec"] = round(len(samples) / max(sample_rate, 1), 2)

            samples = samples.astype(np.float64)
            peak = np.max(np.abs(samples))
            if peak > 0:
                samples = samples / peak

            # 1. Silence ratio
            silence_ratio = float(np.mean(np.abs(samples) < 0.01))
            findings["silence_ratio"] = round(silence_ratio, 3)
            if silence_ratio > 0.7:
                anomalies.append({
                    "type": "Excessive Silence",
                    "description": f"Audio is {silence_ratio:.0%} silence — natural recordings contain ambient noise.",
                    "severity": "medium",
                    "location": {"x": 50, "y": 30},
                })

            # 2. FFT
            if len(samples) > 1024:
                window = samples[:min(len(samples), sample_rate * 5)]
                fft    = np.fft.rfft(window)
                freqs  = np.fft.rfftfreq(len(window), 1.0 / sample_rate)
                mag    = np.abs(fft)
                if len(mag) > 100:
                    idx_8k = int(np.searchsorted(freqs, 8000))
                    if idx_8k < len(mag):
                        hf_ratio = float(np.mean(mag[idx_8k:]) / (np.mean(mag) + 1e-10))
                        findings["hf_energy_ratio"] = round(hf_ratio, 4)
                        if hf_ratio < 0.02 and sample_rate >= 16000:
                            anomalies.append({
                                "type": "Frequency Cutoff",
                                "description": f"Energy above 8kHz is only {hf_ratio:.2%} — TTS often has limited bandwidth.",
                                "severity": "high",
                                "location": {"x": 80, "y": 50},
                            })
                    band_sz = max(len(mag) // 20, 1)
                    bands   = [float(np.mean(mag[i * band_sz:(i + 1) * band_sz])) for i in range(20)]
                    b_mean  = float(np.mean(bands))
                    if b_mean > 0:
                        cv = float(np.std(bands) / b_mean)
                        findings["spectral_cv"] = round(cv, 3)
                        if cv > 2.5:
                            anomalies.append({
                                "type": "Spectral Anomaly",
                                "description": f"Unusual spectral distribution (CV={cv:.2f}) — natural speech has smoother envelope.",
                                "severity": "medium",
                                "location": {"x": 50, "y": 50},
                            })

            # 3. Zero-crossing rate
            if len(samples) > 4096:
                chunk = 2048
                zcrs  = [
                    float(np.sum(np.abs(np.diff(np.sign(samples[s:s + chunk]))) > 0) / chunk)
                    for s in range(0, len(samples) - chunk, chunk)
                ]
                if len(zcrs) > 4:
                    zcr_mean = float(np.mean(zcrs))
                    zcr_std  = float(np.std(zcrs))
                    findings["zcr_mean"] = round(zcr_mean, 4)
                    findings["zcr_std"]  = round(zcr_std, 4)
                    if zcr_std < 0.005 and zcr_mean > 0.05:
                        anomalies.append({
                            "type": "Uniform Zero-Crossing",
                            "description": f"ZCR is unnaturally uniform (std={zcr_std:.4f}) — may indicate speech synthesis.",
                            "severity": "medium",
                            "location": {"x": 30, "y": 70},
                        })

            # 4. Clipping
            clip_ratio = float(np.mean(np.abs(samples) > 0.999))
            findings["clip_ratio"] = round(clip_ratio, 4)
            if clip_ratio > 0.005:
                anomalies.append({
                    "type": "Audio Clipping",
                    "description": f"{clip_ratio:.2%} of samples are clipped — may indicate over-compression or splicing.",
                    "severity": "medium",
                    "location": {"x": 70, "y": 30},
                })

            high_count = sum(1 for a in anomalies if a["severity"] == "high")
            confidence = max(0.2, 0.85 - high_count * 0.15 - len(anomalies) * 0.06)
            return {"anomalies": anomalies, "findings": findings, "confidence": confidence}

        except Exception as e:
            return {"anomalies": [], "findings": {"error": str(e)}, "confidence": 0.5}

    async def _call_claude_audio(self, spectral_data: dict) -> dict | None:
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            return None
        findings = spectral_data.get("findings", {})
        if not findings or findings.get("format") == "unsupported":
            return None

        desc = f"""Audio analysis:
- Format: {findings.get('detected_format', 'N/A')}
- Sample rate: {findings.get('sample_rate', 'N/A')} Hz
- Duration: {findings.get('duration_sec', 'N/A')}s
- Silence ratio: {findings.get('silence_ratio', 'N/A')}
- HF energy (>8kHz): {findings.get('hf_energy_ratio', 'N/A')}
- Spectral CV: {findings.get('spectral_cv', 'N/A')}
- ZCR mean/std: {findings.get('zcr_mean', 'N/A')} / {findings.get('zcr_std', 'N/A')}
- Clip ratio: {findings.get('clip_ratio', 'N/A')}
- Local anomalies: {len(spectral_data.get('anomalies', []))}"""

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                    json={"model": "claude-sonnet-4-20250514", "max_tokens": 1500,
                          "system": self.SYSTEM_PROMPT, "messages": [{"role": "user", "content": desc}]},
                )
                data = resp.json()
                if "content" in data and data["content"]:
                    text = data["content"][0].get("text", "").strip()
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
                            "description": f"Audio identified as synthetic. Method: {parsed.get('likely_method', 'unknown')}.",
                            "severity": "high",
                            "location": {"x": 50, "y": 50},
                        })
                    return {"anomalies": result_anomalies, "confidence": parsed.get("confidence", 0.5), "summary": parsed.get("summary", "")}
        except Exception:
            pass
        return None
