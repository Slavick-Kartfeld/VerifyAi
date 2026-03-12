"""
Copy-Move Detection Agent — זיהוי שכפול אזורים בתמונה.
עובד לוקלית ללא API. משתמש בטכניקת block-matching על DCT features.
"""
import io
import numpy as np
from PIL import Image, ImageFilter


class CopyMoveAgent:
    """זיהוי copy-move forgery באמצעות block matching."""

    AGENT_TYPE = "copy_move"
    BLOCK_SIZE = 16
    SIMILARITY_THRESHOLD = 0.92
    MIN_SHIFT = 30  # מרחק מינימלי בין בלוקים תואמים (פיקסלים)

    async def analyze(self, file_bytes: bytes, filename: str) -> dict:
        try:
            img = Image.open(io.BytesIO(file_bytes)).convert("L")

            # הקטנה לביצועים (max 512px)
            max_dim = 512
            ratio = min(max_dim / img.width, max_dim / img.height, 1.0)
            if ratio < 1.0:
                img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)

            arr = np.array(img, dtype=np.float64)
            matches = self._detect_clones(arr)
            anomalies = []
            confidence = 0.85  # גבוה = נקי

            if matches:
                # מיון לפי quality
                matches.sort(key=lambda m: m["similarity"], reverse=True)
                top = matches[:5]  # מקסימום 5 אזורים

                for i, m in enumerate(top):
                    # המרה לאחוזים
                    x1_pct = int((m["x1"] / arr.shape[1]) * 100)
                    y1_pct = int((m["y1"] / arr.shape[0]) * 100)
                    x2_pct = int((m["x2"] / arr.shape[1]) * 100)
                    y2_pct = int((m["y2"] / arr.shape[0]) * 100)

                    sim = m["similarity"]
                    severity = "high" if sim > 0.97 else "medium" if sim > 0.94 else "low"

                    anomalies.append({
                        "type": "Copy-Move",
                        "description": (
                            f"זוהה אזור משוכפל: בלוק ב-({x1_pct}%,{y1_pct}%) "
                            f"תואם לבלוק ב-({x2_pct}%,{y2_pct}%) "
                            f"ברמת דמיון {sim:.1%}. "
                            f"מרחק הזזה: {m['shift']} פיקסלים."
                        ),
                        "severity": severity,
                        "location": {"x": x1_pct, "y": y1_pct},
                    })

                confidence = max(0.15, 0.85 - len(top) * 0.12)

            return {
                "agent_type": self.AGENT_TYPE,
                "confidence_score": round(confidence, 2),
                "findings": {
                    "clone_regions_found": len(matches) if matches else 0,
                    "top_matches": len(anomalies),
                    "image_size": f"{arr.shape[1]}x{arr.shape[0]}",
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

    def _detect_clones(self, arr: np.ndarray) -> list:
        """Block matching עם DCT-like features."""
        h, w = arr.shape
        bs = self.BLOCK_SIZE

        if h < bs * 3 or w < bs * 3:
            return []

        # חילוץ features לכל בלוק
        blocks = []
        step = bs // 2  # overlap 50%
        for y in range(0, h - bs, step):
            for x in range(0, w - bs, step):
                block = arr[y:y + bs, x:x + bs]
                # Feature vector: mean, std, gradients, corners
                feat = self._block_features(block)
                blocks.append({"x": x, "y": y, "feat": feat})

        if len(blocks) < 10:
            return []

        # מיון לפי feature vector ראשון (lexicographic) לזירוז השוואות
        blocks.sort(key=lambda b: tuple(b["feat"][:4]))

        matches = []
        n = len(blocks)
        window = min(80, n // 4)  # חלון חיפוש

        for i in range(n):
            for j in range(i + 1, min(i + window, n)):
                b1, b2 = blocks[i], blocks[j]

                # בדיקת מרחק מינימלי
                dx = abs(b1["x"] - b2["x"])
                dy = abs(b1["y"] - b2["y"])
                shift = (dx ** 2 + dy ** 2) ** 0.5
                if shift < self.MIN_SHIFT:
                    continue

                # דמיון cosine
                sim = self._cosine_similarity(b1["feat"], b2["feat"])
                if sim >= self.SIMILARITY_THRESHOLD:
                    matches.append({
                        "x1": b1["x"], "y1": b1["y"],
                        "x2": b2["x"], "y2": b2["y"],
                        "similarity": round(sim, 4),
                        "shift": round(shift),
                    })

        # מיזוג אזורים קרובים
        merged = self._merge_nearby(matches)
        return merged

    def _block_features(self, block: np.ndarray) -> np.ndarray:
        """חילוץ feature vector מבלוק."""
        # סטטיסטיקות בסיסיות
        mean = block.mean()
        std = block.std()

        # גרדיאנטים
        gx = np.diff(block, axis=1).mean()
        gy = np.diff(block, axis=0).mean()

        # רגעים (moments) מסדר 2
        cy, cx = np.mgrid[:block.shape[0], :block.shape[1]]
        total = block.sum() + 1e-10
        mx = (cx * block).sum() / total
        my = (cy * block).sum() / total

        # חלוקה ל-4 רבעים
        bs2 = block.shape[0] // 2
        q1 = block[:bs2, :bs2].mean()
        q2 = block[:bs2, bs2:].mean()
        q3 = block[bs2:, :bs2].mean()
        q4 = block[bs2:, bs2:].mean()

        # histogram bins (4)
        hist = np.histogram(block, bins=4, range=(0, 256))[0].astype(float)
        hist /= hist.sum() + 1e-10

        return np.array([mean, std, gx, gy, mx, my, q1, q2, q3, q4, *hist])

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        dot = np.dot(a, b)
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-10 or nb < 1e-10:
            return 0.0
        return dot / (na * nb)

    def _merge_nearby(self, matches: list, radius: int = 20) -> list:
        """מיזוג התאמות קרובות."""
        if not matches:
            return []

        merged = []
        used = set()

        for i, m in enumerate(matches):
            if i in used:
                continue
            group = [m]
            for j, m2 in enumerate(matches[i + 1:], i + 1):
                if j in used:
                    continue
                dist = ((m["x1"] - m2["x1"]) ** 2 + (m["y1"] - m2["y1"]) ** 2) ** 0.5
                if dist < radius:
                    group.append(m2)
                    used.add(j)

            # לקחת את הטוב ביותר מכל קבוצה
            best = max(group, key=lambda g: g["similarity"])
            merged.append(best)
            used.add(i)

        return merged
