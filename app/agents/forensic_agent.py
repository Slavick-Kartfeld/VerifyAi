"""
סוכן פורנזי-טכני — ניתוח ELA, מטאדטה EXIF, ודחיסה.
עובד באמת על תמונות בלי צורך ב-API חיצוני.
"""
import io
import json
import hashlib
from datetime import datetime
from PIL import Image, ImageChops, ExifTags
import numpy as np


class ForensicTechnicalAgent:
    """סוכן לניתוח פורנזי-טכני: ELA, מטאדטה, גרעון, דחיסה"""

    AGENT_TYPE = "forensic_technical"

    async def analyze(self, file_bytes: bytes, filename: str) -> dict:
        anomalies = []
        findings = {}

        try:
            img = Image.open(io.BytesIO(file_bytes))
        except Exception:
            return self._result(0.5, findings, [
                self._anomaly("format", "לא ניתן לפתוח את הקובץ כתמונה", "medium")
            ])

        # 1. ניתוח מטאדטה EXIF
        exif_anomalies, exif_findings = self._analyze_exif(img)
        anomalies.extend(exif_anomalies)
        findings["exif"] = exif_findings

        # 2. ניתוח ELA (Error Level Analysis)
        ela_anomalies, ela_findings = self._analyze_ela(img)
        anomalies.extend(ela_anomalies)
        findings["ela"] = ela_findings

        # 3. ניתוח דחיסה
        comp_anomalies, comp_findings = self._analyze_compression(file_bytes, img)
        anomalies.extend(comp_anomalies)
        findings["compression"] = comp_findings

        # 4. ניתוח מימדים וגרעון
        dim_anomalies, dim_findings = self._analyze_dimensions(img)
        anomalies.extend(dim_anomalies)
        findings["dimensions"] = dim_findings

        # חישוב ציון ביטחון
        confidence = self._calculate_confidence(anomalies)

        return self._result(confidence, findings, anomalies)

    def _analyze_exif(self, img: Image.Image) -> tuple:
        anomalies = []
        findings = {}

        exif_data = {}
        if hasattr(img, '_getexif') and img._getexif():
            raw_exif = img._getexif()
            for tag_id, value in raw_exif.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                if isinstance(value, bytes):
                    try:
                        value = value.decode('utf-8', errors='ignore')
                    except Exception:
                        value = str(value)[:100]
                exif_data[str(tag)] = str(value)[:200]

        findings["has_exif"] = bool(exif_data)
        findings["exif_tags_count"] = len(exif_data)

        if not exif_data:
            anomalies.append(self._anomaly(
                "מטאדטה",
                "הקובץ נעדר מטאדטה EXIF לחלוטין. ייתכן שהמטאדטה הוסרה בכוונה כדי להסתיר את מקור התמונה.",
                "medium",
                {"x": 90, "y": 10}
            ))
        else:
            # בדיקת תוכנת עריכה
            software = exif_data.get("Software", "")
            if software:
                findings["software"] = software
                editing_tools = ["photoshop", "gimp", "lightroom", "snapseed", "picsart", "canva"]
                for tool in editing_tools:
                    if tool.lower() in software.lower():
                        anomalies.append(self._anomaly(
                            "תוכנת עריכה",
                            f"זוהתה תוכנת עריכה במטאדטה: {software}. התמונה עברה עיבוד.",
                            "medium",
                            {"x": 85, "y": 8}
                        ))
                        break

            # בדיקת תאריכים
            date_original = exif_data.get("DateTimeOriginal", "")
            date_modified = exif_data.get("DateTime", "")
            if date_original and date_modified and date_original != date_modified:
                findings["date_original"] = date_original
                findings["date_modified"] = date_modified
                anomalies.append(self._anomaly(
                    "תאריכים",
                    f"פער בין תאריך הצילום ({date_original}) לתאריך העריכה ({date_modified}).",
                    "high",
                    {"x": 80, "y": 15}
                ))

        return anomalies, findings

    def _analyze_ela(self, img: Image.Image) -> tuple:
        anomalies = []
        findings = {}

        try:
            # שמירה מחדש ב-JPEG באיכות 95 והשוואה
            if img.mode != 'RGB':
                img = img.convert('RGB')

            buffer = io.BytesIO()
            img.save(buffer, 'JPEG', quality=95)
            buffer.seek(0)
            resaved = Image.open(buffer)

            # חישוב ההפרש
            ela_img = ImageChops.difference(img, resaved)
            ela_array = np.array(ela_img, dtype=np.float64)

            # סטטיסטיקות
            mean_error = float(np.mean(ela_array))
            max_error = float(np.max(ela_array))
            std_error = float(np.std(ela_array))

            findings["mean_error"] = round(mean_error, 2)
            findings["max_error"] = round(max_error, 2)
            findings["std_error"] = round(std_error, 2)

            # חלוקה לאזורים (grid 4x4) לזיהוי אנומליות מקומיות
            h, w = ela_array.shape[:2]
            grid_h, grid_w = h // 4, w // 4
            region_means = []

            for row in range(4):
                for col in range(4):
                    region = ela_array[row*grid_h:(row+1)*grid_h, col*grid_w:(col+1)*grid_w]
                    region_means.append({
                        "row": row, "col": col,
                        "mean": float(np.mean(region))
                    })

            overall_mean = np.mean([r["mean"] for r in region_means])
            findings["region_analysis"] = True

            # זיהוי אזורים חריגים
            for r in region_means:
                if r["mean"] > overall_mean * 2.5 and r["mean"] > 15:
                    x_pct = int((r["col"] * 25) + 12)
                    y_pct = int((r["row"] * 25) + 12)
                    ratio = round(r["mean"] / overall_mean, 1)
                    anomalies.append(self._anomaly(
                        "ELA",
                        f"חריגה ברמת שגיאת ELA באזור ({r['row']+1},{r['col']+1}). "
                        f"עוצמת השגיאה גבוהה פי {ratio} מהממוצע, "
                        f"המעידה על עריכה או הדבקה באזור זה.",
                        "high",
                        {"x": x_pct, "y": y_pct}
                    ))

            # אם השונות הכללית גבוהה מאוד
            if std_error > 20:
                anomalies.append(self._anomaly(
                    "ELA כללי",
                    f"שונות גבוהה ברמת ה-ELA ({round(std_error, 1)}). "
                    f"מעידה על רמות דחיסה שונות בחלקים שונים של התמונה.",
                    "medium",
                    {"x": 50, "y": 50}
                ))

        except Exception as e:
            findings["ela_error"] = str(e)

        return anomalies, findings

    def _analyze_compression(self, file_bytes: bytes, img: Image.Image) -> tuple:
        anomalies = []
        findings = {}

        findings["format"] = img.format or "unknown"
        findings["file_size_bytes"] = len(file_bytes)
        findings["mode"] = img.mode

        if img.format == "JPEG":
            # בדיקת quantization tables
            if hasattr(img, 'quantization') and img.quantization:
                tables = img.quantization
                findings["quantization_tables"] = len(tables)

                # double JPEG compression detection
                if len(tables) > 0:
                    table_values = list(tables.values())[0]
                    if isinstance(table_values, (list, tuple)):
                        q_std = float(np.std(table_values))
                        findings["q_table_std"] = round(q_std, 2)

                        if q_std > 25:
                            anomalies.append(self._anomaly(
                                "דחיסה כפולה",
                                "נמצאו סימנים לדחיסת JPEG כפולה (double compression). "
                                "זה עלול להעיד על שמירה מחדש לאחר עריכה.",
                                "medium",
                                {"x": 50, "y": 85}
                            ))

        # בדיקת יחס גודל קובץ למימדים
        pixels = img.width * img.height
        if pixels > 0:
            bytes_per_pixel = len(file_bytes) / pixels
            findings["bytes_per_pixel"] = round(bytes_per_pixel, 4)

            if bytes_per_pixel < 0.1 and img.format == "JPEG":
                anomalies.append(self._anomaly(
                    "דחיסה",
                    f"יחס דחיסה חריג ({round(bytes_per_pixel, 3)} bytes/pixel). "
                    f"התמונה דחוסה מאוד ביחס למימדיה, ייתכן שנשמרה מספר פעמים.",
                    "low",
                    {"x": 15, "y": 90}
                ))

        return anomalies, findings

    def _analyze_dimensions(self, img: Image.Image) -> tuple:
        anomalies = []
        findings = {
            "width": img.width,
            "height": img.height,
            "megapixels": round((img.width * img.height) / 1e6, 2)
        }

        # תמונות AI נוטות למימדים עגולים
        ai_dimensions = [
            (512, 512), (768, 768), (1024, 1024), (1024, 1792), (1792, 1024),
            (512, 768), (768, 512), (1024, 768), (768, 1024)
        ]
        if (img.width, img.height) in ai_dimensions:
            anomalies.append(self._anomaly(
                "מימדים",
                f"מימדי התמונה ({img.width}x{img.height}) תואמים לפלט אופייני של מודלי AI "
                f"(כגון DALL-E, Midjourney, Stable Diffusion).",
                "medium",
                {"x": 10, "y": 10}
            ))

        return anomalies, findings

    def _calculate_confidence(self, anomalies: list) -> float:
        if not anomalies:
            return 0.92  # אין אנומליות — ביטחון גבוה באותנטיות

        severity_scores = {"high": 0.15, "medium": 0.08, "low": 0.03}
        total_penalty = sum(severity_scores.get(a["severity"], 0) for a in anomalies)
        return max(0.15, round(0.92 - total_penalty, 2))

    def _anomaly(self, type_: str, desc: str, severity: str, location: dict = None) -> dict:
        a = {"type": type_, "description": desc, "severity": severity}
        if location:
            a["location"] = location
        return a

    def _result(self, confidence: float, findings: dict, anomalies: list) -> dict:
        return {
            "agent_type": self.AGENT_TYPE,
            "confidence_score": confidence,
            "findings": findings,
            "anomalies": {"items": anomalies}
        }
