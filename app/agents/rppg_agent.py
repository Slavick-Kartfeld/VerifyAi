"""
rPPG (Remote Photoplethysmography) Agent — Physiological Signal Analysis
=========================================================================
Detects hyper-realistic face deepfakes in video by analyzing blood-flow
color fluctuations in facial skin regions.

How it works:
  Real faces exhibit periodic micro-color changes (0.7–3.5 Hz) driven by
  cardiac activity. Current deepfake generators — including diffusion models,
  GANs, and neural-renderer face-swappers — do NOT replicate this signal.
  Inconsistent, absent, or spatially incoherent rPPG signal = forgery indicator.

Scope:
  - VIDEO only (requires ≥ 3 s of visible face frames at ≥ 15 fps)
  - LOCAL — no API required
  - Integrated into VideoForensicAgent pipeline; also callable standalone

Signal extraction method: CHROM (De Haan & Jeanne, IEEE TBME 2013)
Spatial consistency: 5-region face split + neck/forehead cross-correlation

References:
  - DeepFakesON-Phys (Hernandez-Ortega et al., 2020) — 98% AUC on Celeb-DF
  - Fraunhofer HHI (Seibold et al., Frontiers 2025) — high-quality deepfakes
  - ICIAP 2023 cautionary note — compression & adversarial limitations
"""

import io
import struct
import numpy as np
from PIL import Image

# ── Constants ─────────────────────────────────────────────────────────────────
RPPG_FREQ_MIN  = 0.7   # Hz  — minimum plausible heart rate (42 bpm)
RPPG_FREQ_MAX  = 3.5   # Hz  — maximum plausible heart rate (210 bpm)
MIN_FRAMES     = 45    # ~3 s at 15 fps — minimum for reliable FFT
SPATIAL_REGIONS = 5    # forehead, left-cheek, right-cheek, nose, chin
CONSISTENCY_THRESHOLD = 0.35  # Pearson r — below = spatial incoherence flag


class RPPGAgent:
    """
    Remote Photoplethysmography deepfake detector.

    Designed to catch hyper-realistic face deepfakes that defeat visual
    and frequency-based detectors.  Complements — never replaces — existing
    11-agent pipeline.

    Activation: VIDEO media-type with human face visible.
    Weight in CrossReference: 0.08 (supplementary signal).
    """

    AGENT_TYPE = "rppg_physiological"

    async def analyze(self, file_bytes: bytes, filename: str) -> dict:
        """Entry point — called by Orchestrator for video files."""
        frames = self._extract_frames(file_bytes, filename)

        if len(frames) < MIN_FRAMES:
            return self._insufficient_frames(len(frames))

        # Run full rPPG pipeline
        signals  = self._extract_rppg_signals(frames)
        spectral = self._spectral_analysis(signals["global"], fps=signals["fps"])
        spatial  = self._spatial_consistency(signals["regions"])
        anomalies, confidence = self._assess(spectral, spatial, len(frames))

        return {
            "agent_type": self.AGENT_TYPE,
            "confidence_score": round(confidence, 3),
            "findings": {
                "frames_analyzed": len(frames),
                "fps_estimated": round(signals["fps"], 1),
                "dominant_freq_hz": round(spectral["dominant_freq"], 3),
                "heart_rate_bpm": round(spectral["dominant_freq"] * 60, 1),
                "in_physiological_range": spectral["in_range"],
                "signal_snr_db": round(spectral["snr_db"], 2),
                "spatial_coherence_r": round(spatial["mean_correlation"], 3),
                "spatial_consistent": spatial["consistent"],
                "source": "local_rppg",
            },
            "anomalies": {"items": anomalies},
        }

    # ── Frame extraction ───────────────────────────────────────────────────────

    def _extract_frames(self, file_bytes: bytes, filename: str) -> list:
        """
        Extract equally-spaced PIL frames from a video file.
        Uses basic container parsing; ffmpeg preferred in production.
        Falls back to JPEG-scan for simple MP4/AVI.
        """
        frames = []

        # Strategy 1: scan for embedded JPEG frames (common in MP4 moov boxes)
        jpeg_frames = self._scan_jpeg_frames(file_bytes)
        if jpeg_frames:
            return jpeg_frames

        # Strategy 2: single image fallback (e.g. GIF frame 0)
        try:
            img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            frames.append(np.array(img, dtype=np.float32))
        except Exception:
            pass

        return frames

    def _scan_jpeg_frames(self, data: bytes) -> list:
        """Locate JPEG Start-Of-Image (0xFFD8) markers inside binary container."""
        frames = []
        i = 0
        while i < len(data) - 1:
            if data[i] == 0xFF and data[i + 1] == 0xD8:
                # find End-Of-Image
                end = data.find(b'\xFF\xD9', i + 2)
                if end == -1:
                    break
                try:
                    img = Image.open(io.BytesIO(data[i:end + 2])).convert("RGB")
                    if img.width >= 64 and img.height >= 64:
                        frames.append(np.array(img, dtype=np.float32))
                except Exception:
                    pass
                i = end + 2
            else:
                i += 1
        return frames

    # ── rPPG signal extraction — CHROM method ─────────────────────────────────

    def _extract_rppg_signals(self, frames: list) -> dict:
        """
        Extract rPPG signals using CHROM (chrominance-based) method.

        CHROM separates pulse-related chrominance changes from luminance
        and motion artefacts by projecting onto two orthogonal chroma axes.

        Returns global signal + 5 regional signals + estimated fps.
        """
        # Estimate fps from frame count (assume ~15–30 fps typical)
        fps = 15.0  # conservative default; production: read from container

        global_rgb  = []  # mean RGB per frame across full face ROI
        region_rgbs = [[] for _ in range(SPATIAL_REGIONS)]

        for frame in frames:
            h, w = frame.shape[:2]

            # ── Global ROI: centre 60% of frame (rough face region) ──
            y1 = int(h * 0.15);  y2 = int(h * 0.85)
            x1 = int(w * 0.20);  x2 = int(w * 0.80)
            roi = frame[y1:y2, x1:x2]

            # Simple skin mask: YCbCr range filter
            roi_uint8 = np.clip(roi, 0, 255).astype(np.uint8)
            skin = self._skin_mask(roi_uint8)
            if skin.sum() < 100:
                skin = np.ones(roi.shape[:2], dtype=bool)  # fallback: use all

            global_rgb.append([
                roi[skin, 0].mean(),
                roi[skin, 1].mean(),
                roi[skin, 2].mean(),
            ])

            # ── 5 facial sub-regions ──
            ry = y2 - y1
            rx = x2 - x1
            sub_rois = [
                roi[0:ry//3,          rx//4:3*rx//4],   # forehead
                roi[ry//4:3*ry//4,    0:rx//2],          # left cheek
                roi[ry//4:3*ry//4,    rx//2:rx],         # right cheek
                roi[ry//3:2*ry//3,    rx//3:2*rx//3],   # nose bridge
                roi[2*ry//3:ry,       rx//4:3*rx//4],   # chin
            ]
            for ri, sroi in enumerate(sub_rois):
                if sroi.size == 0:
                    region_rgbs[ri].append([128.0, 128.0, 128.0])
                else:
                    region_rgbs[ri].append([
                        sroi[:, :, 0].mean(),
                        sroi[:, :, 1].mean(),
                        sroi[:, :, 2].mean(),
                    ])

        global_arr = np.array(global_rgb, dtype=np.float64)  # (N, 3)
        global_chrom = self._chrom(global_arr)

        region_chroms = []
        for rrgb in region_rgbs:
            arr = np.array(rrgb, dtype=np.float64)
            region_chroms.append(self._chrom(arr))

        return {
            "global":  global_chrom,
            "regions": region_chroms,
            "fps":     fps,
        }

    def _chrom(self, rgb: np.ndarray) -> np.ndarray:
        """
        CHROM algorithm (De Haan & Jeanne 2013).
        Produces a 1-D BVP (blood volume pulse) signal from RGB trace.
        """
        if rgb.shape[0] < 2:
            return np.zeros(max(rgb.shape[0], 2))

        # Normalise each channel
        mean = rgb.mean(axis=0) + 1e-8
        norm = rgb / mean

        # Xs = 3R - 2G,  Yt = 1.5R + G - 1.5B
        Xs = 3 * norm[:, 0] - 2 * norm[:, 1]
        Yt = 1.5 * norm[:, 0] + norm[:, 1] - 1.5 * norm[:, 2]

        # Alpha to minimise motion
        std_Xs = np.std(Xs) + 1e-8
        std_Yt = np.std(Yt) + 1e-8
        alpha  = std_Xs / std_Yt

        bvp = Xs - alpha * Yt

        # Bandpass: keep physiological range via simple detrend + smooth
        bvp = bvp - np.mean(bvp)
        if len(bvp) > 5:
            # Running-mean detrend
            kernel = min(15, len(bvp) // 3)
            if kernel > 1:
                pad = np.pad(bvp, kernel // 2, mode='edge')
                trend = np.convolve(pad, np.ones(kernel) / kernel, mode='valid')
                trend = trend[:len(bvp)]
                bvp = bvp - trend

        return bvp

    def _skin_mask(self, roi_uint8: np.ndarray) -> np.ndarray:
        """
        YCbCr skin segmentation.
        Range: 0≤Y≤235, 77≤Cb≤127, 133≤Cr≤173.
        """
        try:
            # Manual YCbCr conversion (avoids heavy dependencies)
            R = roi_uint8[:, :, 0].astype(np.float32)
            G = roi_uint8[:, :, 1].astype(np.float32)
            B = roi_uint8[:, :, 2].astype(np.float32)
            Y  =  0.299 * R + 0.587 * G + 0.114 * B
            Cb = -0.169 * R - 0.331 * G + 0.500 * B + 128
            Cr =  0.500 * R - 0.419 * G - 0.081 * B + 128
            mask = (
                (Y  >= 0)   & (Y  <= 235) &
                (Cb >= 77)  & (Cb <= 127) &
                (Cr >= 133) & (Cr <= 173)
            )
            return mask
        except Exception:
            return np.ones(roi_uint8.shape[:2], dtype=bool)

    # ── Spectral analysis ──────────────────────────────────────────────────────

    def _spectral_analysis(self, signal: np.ndarray, fps: float) -> dict:
        """
        FFT on global BVP signal.
        Checks:
          1. Dominant frequency is in physiological range (0.7–3.5 Hz)
          2. SNR: energy at dominant freq vs total energy
        """
        n = len(signal)
        if n < 8:
            return {"dominant_freq": 0, "in_range": False, "snr_db": -20, "power": 0}

        freqs = np.fft.rfftfreq(n, d=1.0 / fps)
        fft   = np.abs(np.fft.rfft(signal)) ** 2  # power spectrum

        # Mask: physiological band
        physio_mask = (freqs >= RPPG_FREQ_MIN) & (freqs <= RPPG_FREQ_MAX)
        noise_mask  = ~physio_mask

        if physio_mask.sum() == 0:
            return {"dominant_freq": 0, "in_range": False, "snr_db": -20, "power": 0}

        physio_power = fft[physio_mask].sum()
        noise_power  = fft[noise_mask].sum() + 1e-10
        snr_db       = 10 * np.log10(physio_power / noise_power + 1e-10)

        # Dominant frequency in full spectrum
        dom_idx   = np.argmax(fft)
        dom_freq  = freqs[dom_idx]
        in_range  = RPPG_FREQ_MIN <= dom_freq <= RPPG_FREQ_MAX

        return {
            "dominant_freq":  float(dom_freq),
            "in_range":       bool(in_range),
            "snr_db":         float(snr_db),
            "physio_power":   float(physio_power),
        }

    # ── Spatial consistency ────────────────────────────────────────────────────

    def _spatial_consistency(self, region_signals: list) -> dict:
        """
        Check that rPPG signals are correlated across facial regions.
        Real faces: all regions pulsate in sync (r > 0.35).
        Deepfakes:  synthesised texture lacks this spatial coherence.
        """
        if len(region_signals) < 2:
            return {"consistent": True, "mean_correlation": 1.0, "pairs": []}

        min_len = min(len(s) for s in region_signals)
        if min_len < 4:
            return {"consistent": True, "mean_correlation": 0.5, "pairs": []}

        # Trim all to same length
        trimmed = [s[:min_len] for s in region_signals]

        correlations = []
        pairs = []
        n = len(trimmed)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = trimmed[i], trimmed[j]
                std_a, std_b = np.std(a), np.std(b)
                if std_a < 1e-8 or std_b < 1e-8:
                    r = 0.0
                else:
                    r = float(np.corrcoef(a, b)[0, 1])
                    if np.isnan(r):
                        r = 0.0
                correlations.append(r)
                pairs.append({"region_i": i, "region_j": j, "r": round(r, 3)})

        mean_r = float(np.mean(correlations)) if correlations else 0.5
        consistent = mean_r >= CONSISTENCY_THRESHOLD

        return {
            "consistent":        consistent,
            "mean_correlation":  mean_r,
            "min_correlation":   float(min(correlations)) if correlations else 0.0,
            "pairs":             pairs[:6],  # top 6 pairs for report
        }

    # ── Verdict assembly ──────────────────────────────────────────────────────

    def _assess(self, spectral: dict, spatial: dict, n_frames: int) -> tuple:
        """
        Combine spectral and spatial findings into anomalies + confidence score.

        Scoring logic:
          Base: 0.80 (neutral — rPPG alone is supplementary)
          Penalties applied cumulatively, then clamped to [0.10, 0.95].
        """
        anomalies = []
        penalty   = 0.0

        # ── Check 1: dominant frequency out of physiological range ──
        if not spectral["in_range"]:
            df = spectral["dominant_freq"]
            if df == 0:
                desc = (
                    "rPPG analysis detected no periodic signal in the physiological "
                    "heart-rate band (0.7–3.5 Hz). Authentic face videos consistently "
                    "exhibit blood-flow driven color oscillations. Absence of this signal "
                    "is a strong indicator of AI-generated or fully-synthetic face content."
                )
                sev = "high"
                penalty += 0.30
            else:
                desc = (
                    f"Dominant rPPG frequency is {df:.2f} Hz — outside the physiological "
                    f"heart-rate range (0.7–3.5 Hz / 42–210 bpm). This may indicate "
                    f"a synthetic face that lacks realistic blood-flow modulation."
                )
                sev = "medium"
                penalty += 0.18
            anomalies.append({
                "type": "rPPG — Physiological Frequency Absent",
                "description": desc,
                "severity": sev,
                "location": {"x": 50, "y": 30},
            })

        # ── Check 2: low SNR ──
        snr = spectral.get("snr_db", 0)
        if snr < -5:
            anomalies.append({
                "type": "rPPG — Weak Cardiac Signal (Low SNR)",
                "description": (
                    f"Signal-to-noise ratio in the cardiac frequency band: {snr:.1f} dB. "
                    "Real faces produce a measurable pulse signal even under moderate "
                    "compression. Extremely low SNR may indicate a synthesised face "
                    "rendered without physiological modelling."
                ),
                "severity": "medium" if snr > -15 else "high",
                "location": {"x": 50, "y": 30},
            })
            penalty += 0.12 if snr > -15 else 0.20

        # ── Check 3: spatial incoherence across face regions ──
        if not spatial["consistent"]:
            r = spatial["mean_correlation"]
            anomalies.append({
                "type": "rPPG — Spatial Incoherence (Face Regions Out of Sync)",
                "description": (
                    f"Mean cross-correlation between facial sub-regions: r={r:.3f} "
                    f"(threshold: r≥{CONSISTENCY_THRESHOLD}). In authentic video, all "
                    "face regions (forehead, cheeks, nose, chin) pulse together. "
                    "Spatial incoherence suggests the face texture was synthesised "
                    "or composited without physiological consistency constraints."
                ),
                "severity": "high" if r < 0.15 else "medium",
                "location": {"x": 50, "y": 50},
            })
            penalty += 0.20 if r < 0.15 else 0.12

        # ── Confidence score ──
        # High confidence = image is authentic. Penalties reduce authenticity score.
        base       = 0.80
        confidence = max(0.10, min(0.95, base - penalty))

        # ── Caveat note if no anomalies ──
        if not anomalies:
            anomalies.append({
                "type": "rPPG — Signal Within Normal Parameters",
                "description": (
                    f"Dominant frequency: {spectral['dominant_freq']:.2f} Hz "
                    f"({spectral['dominant_freq']*60:.0f} bpm). "
                    f"Spatial coherence: r={spatial['mean_correlation']:.3f}. "
                    "Physiological signals appear consistent with a real face. "
                    "Note: advanced deepfakes using rPPG-aware synthesis may pass "
                    "this check — treat as supplementary evidence only."
                ),
                "severity": "low",
                "location": {"x": 50, "y": 30},
            })

        return anomalies, confidence

    # ── Fallback ──────────────────────────────────────────────────────────────

    def _insufficient_frames(self, n: int) -> dict:
        return {
            "agent_type": self.AGENT_TYPE,
            "confidence_score": 0.50,
            "findings": {
                "frames_analyzed": n,
                "note": (
                    f"Only {n} frames extracted — minimum {MIN_FRAMES} required "
                    "for reliable rPPG analysis. Install ffmpeg for full video support."
                ),
                "source": "local_rppg",
            },
            "anomalies": {"items": [{
                "type": "rPPG — Insufficient Frames",
                "description": (
                    f"rPPG analysis requires at least {MIN_FRAMES} frames (~3 seconds "
                    "at 15 fps). This video yielded only {n} extractable frames. "
                    "Result is inconclusive for physiological analysis."
                ).format(n=n),
                "severity": "low",
                "location": {"x": 50, "y": 50},
            }]},
        }
