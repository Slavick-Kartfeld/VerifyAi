"""
Frequency Analysis Agent — ניתוח תדירותי של תמונות.
עובד לוקלית ללא API. מזהה:
- חריגות בספקטרום DCT (תמונות AI/מעובדות)
- JPEG grid misalignment (הרכבה מתמונות שונות)
- High-frequency anomalies (החלקה/שכפול מקומי)
"""
import io
import numpy as np
from PIL import Image


class FrequencyAnalysisAgent:
    """ניתוח תדירותי פורנזי."""

    AGENT_TYPE = "frequency_analysis"

    async def analyze(self, file_bytes: bytes, filename: str) -> dict:
        try:
            img = Image.open(io.BytesIO(file_bytes)).convert("L")

            # Resize for performance
            max_dim = 512
            ratio = min(max_dim / img.width, max_dim / img.height, 1.0)
            if ratio < 1.0:
                img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)

            arr = np.array(img, dtype=np.float64)
            anomalies = []

            # 1. DCT spectral analysis
            dct_result = self._analyze_dct_spectrum(arr)
            anomalies.extend(dct_result["anomalies"])

            # 2. JPEG grid consistency
            grid_result = self._check_jpeg_grid(arr)
            anomalies.extend(grid_result["anomalies"])

            # 3. High-frequency energy map
            hf_result = self._high_frequency_map(arr)
            anomalies.extend(hf_result["anomalies"])

            # 4. Noise consistency
            noise_result = self._noise_consistency(arr)
            anomalies.extend(noise_result["anomalies"])

            # Calculate confidence
            if not anomalies:
                confidence = 0.88
            else:
                high_count = sum(1 for a in anomalies if a["severity"] == "high")
                confidence = max(0.15, 0.88 - high_count * 0.15 - len(anomalies) * 0.05)

            return {
                "agent_type": self.AGENT_TYPE,
                "confidence_score": round(confidence, 2),
                "findings": {
                    "dct_score": dct_result.get("score", 0),
                    "grid_aligned": grid_result.get("aligned", True),
                    "hf_anomaly_regions": hf_result.get("regions", 0),
                    "noise_uniform": noise_result.get("uniform", True),
                    "source": "local_analysis",
                },
                "anomalies": {"items": anomalies},
            }

        except Exception as e:
            return {
                "agent_type": self.AGENT_TYPE,
                "confidence_score": 0.5,
                "findings": {"error": str(e), "source": "local_analysis"},
                "anomalies": {"items": []},
            }

    def _analyze_dct_spectrum(self, arr: np.ndarray) -> dict:
        """ניתוח ספקטרום DCT — תמונות AI יש להן חתימה ייחודית."""
        anomalies = []
        h, w = arr.shape

        # Use FFT as DCT proxy (numpy has FFT built-in)
        f_transform = np.fft.fft2(arr)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.log1p(np.abs(f_shift))

        # Analyze radial energy distribution
        cy, cx = h // 2, w // 2
        max_radius = min(cy, cx)

        radial_energy = []
        for r in range(1, max_radius, max(1, max_radius // 32)):
            y, x = np.ogrid[-cy:h - cy, -cx:w - cx]
            mask = (x * x + y * y >= r * r) & (x * x + y * y < (r + max_radius // 32) ** 2)
            if mask.any():
                radial_energy.append(magnitude[mask].mean())

        if len(radial_energy) >= 8:
            # Check for unnatural spectral decay
            energy = np.array(radial_energy)
            # Natural images have smooth log-linear decay
            # AI images often have periodic peaks or sudden drops

            # Check variance of differences (should be smooth)
            diffs = np.diff(energy)
            diff_std = np.std(diffs)
            diff_mean = np.abs(np.mean(diffs))

            if diff_mean > 0:
                irregularity = diff_std / (diff_mean + 1e-10)
            else:
                irregularity = 0

            score = round(min(irregularity / 3.0, 1.0), 3)

            if irregularity > 2.5:
                anomalies.append({
                    "type": "Spectral Irregularity",
                    "description": f"Frequency spectrum shows irregular energy decay (score: {score:.2f}). "
                                 f"Natural images typically have smooth spectral falloff. "
                                 f"This pattern may indicate AI generation or heavy processing.",
                    "severity": "high" if irregularity > 3.5 else "medium",
                    "location": {"x": 50, "y": 50},
                })
            elif irregularity > 1.8:
                anomalies.append({
                    "type": "Spectral Pattern",
                    "description": f"Minor spectral irregularity detected (score: {score:.2f}). "
                                 f"May indicate post-processing or compression artifacts.",
                    "severity": "low",
                    "location": {"x": 50, "y": 50},
                })

            return {"anomalies": anomalies, "score": score}

        return {"anomalies": [], "score": 0}

    def _check_jpeg_grid(self, arr: np.ndarray) -> dict:
        """בדיקת יישור JPEG grid — הרכבה תשבור את ה-8x8 grid."""
        anomalies = []
        h, w = arr.shape

        if h < 64 or w < 64:
            return {"anomalies": [], "aligned": True}

        # Check 8x8 block boundary energy
        # In JPEG images, block boundaries at multiples of 8 have specific artifacts
        block_scores = []

        for offset in range(8):
            # Measure vertical edge strength at this offset
            edges = 0
            count = 0
            for x in range(offset, w - 1, 8):
                if x + 1 < w:
                    diff = np.abs(arr[:, x].astype(float) - arr[:, x + 1].astype(float))
                    edges += diff.mean()
                    count += 1
            if count > 0:
                block_scores.append(edges / count)

        if len(block_scores) == 8:
            scores = np.array(block_scores)
            primary = np.argmax(scores)
            peak_ratio = scores[primary] / (np.mean(scores) + 1e-10)

            # In a normal JPEG, offset 7 (boundary at 8) should have highest energy
            # If a different offset wins, the image may have been cropped/spliced
            expected_peak = 7  # 0-indexed position for 8-pixel grid

            aligned = primary == expected_peak or peak_ratio < 1.15

            if not aligned and peak_ratio > 1.3:
                anomalies.append({
                    "type": "JPEG Grid Misalignment",
                    "description": f"JPEG 8x8 block grid is misaligned (offset {primary} vs expected {expected_peak}). "
                                 f"Peak ratio: {peak_ratio:.2f}. This suggests the image was cropped after JPEG "
                                 f"compression or spliced from another image.",
                    "severity": "high" if peak_ratio > 1.6 else "medium",
                    "location": {"x": 50, "y": 10},
                })
                return {"anomalies": anomalies, "aligned": False}

        return {"anomalies": anomalies, "aligned": True}

    def _high_frequency_map(self, arr: np.ndarray) -> dict:
        """מפת אנרגיה בתדירויות גבוהות — אזורים מטושטשים/מועתקים חסרים HF."""
        anomalies = []
        h, w = arr.shape

        if h < 64 or w < 64:
            return {"anomalies": [], "regions": 0}

        # Compute high-frequency energy using Laplacian
        # Laplacian = second derivative, captures edges/texture
        kernel_size = 3
        padded = np.pad(arr, 1, mode='reflect')
        laplacian = (
            padded[:-2, 1:-1] + padded[2:, 1:-1] +
            padded[1:-1, :-2] + padded[1:-1, 2:] -
            4 * padded[1:-1, 1:-1]
        )

        # Divide into blocks and check variance
        block_size = max(32, min(h, w) // 8)
        variances = []
        positions = []

        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = laplacian[y:y + block_size, x:x + block_size]
                var = np.var(block)
                variances.append(var)
                positions.append((x, y))

        if len(variances) < 4:
            return {"anomalies": [], "regions": 0}

        variances = np.array(variances)
        median_var = np.median(variances)
        anomaly_regions = 0

        for i, (var, (x, y)) in enumerate(zip(variances, positions)):
            if median_var > 0:
                ratio = var / median_var
                # Very low HF energy compared to surroundings = smoothed/inpainted
                if ratio < 0.15 and median_var > 5:
                    x_pct = int((x + block_size / 2) / w * 100)
                    y_pct = int((y + block_size / 2) / h * 100)
                    anomaly_regions += 1
                    if anomaly_regions <= 3:  # Max 3 reported
                        anomalies.append({
                            "type": "Low-Frequency Region",
                            "description": f"Abnormally smooth region at ({x_pct}%,{y_pct}%) — "
                                         f"high-frequency energy is {ratio:.0%} of image median. "
                                         f"May indicate inpainting, cloning, or AI-generated fill.",
                            "severity": "medium",
                            "location": {"x": x_pct, "y": y_pct},
                        })

        return {"anomalies": anomalies, "regions": anomaly_regions}

    def _noise_consistency(self, arr: np.ndarray) -> dict:
        """בדיקת עקביות רעש — אזורים שונים צריכים להראות רעש דומה."""
        anomalies = []
        h, w = arr.shape

        if h < 64 or w < 64:
            return {"anomalies": [], "uniform": True}

        # Extract noise by subtracting smoothed version
        from PIL import ImageFilter
        img = Image.fromarray(arr.astype(np.uint8))
        smooth = np.array(img.filter(ImageFilter.GaussianBlur(2)), dtype=np.float64)
        noise = arr - smooth

        # Divide into quadrants and compare noise statistics
        qh, qw = h // 2, w // 2
        quads = [
            noise[:qh, :qw],      # top-left
            noise[:qh, qw:],      # top-right
            noise[qh:, :qw],      # bottom-left
            noise[qh:, qw:],      # bottom-right
        ]
        quad_names = ["top-left", "top-right", "bottom-left", "bottom-right"]
        quad_pos = [{"x": 25, "y": 25}, {"x": 75, "y": 25}, {"x": 25, "y": 75}, {"x": 75, "y": 75}]

        stds = [np.std(q) for q in quads]
        mean_std = np.mean(stds)

        uniform = True
        if mean_std > 0.5:  # Skip near-flat images
            for i, (std, name, pos) in enumerate(zip(stds, quad_names, quad_pos)):
                ratio = std / mean_std if mean_std > 0 else 1
                if ratio < 0.4 or ratio > 2.2:
                    uniform = False
                    anomalies.append({
                        "type": "Noise Inconsistency",
                        "description": f"Noise level in {name} quadrant ({std:.1f}) differs significantly "
                                     f"from image average ({mean_std:.1f}). Ratio: {ratio:.2f}. "
                                     f"Different noise profiles suggest compositing from multiple sources.",
                        "severity": "high" if (ratio < 0.25 or ratio > 3.0) else "medium",
                        "location": pos,
                    })

        return {"anomalies": anomalies, "uniform": uniform}
