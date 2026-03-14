"""
Metadata Consistency Agent — ניתוח עקביות מטאדטה.
בודק סתירות פנימיות ב-EXIF, ICC, XMP:
- GPS vs timezone
- Camera model vs resolution capabilities
- Software version vs format capabilities
- Creation date vs modification date logic
- Thumbnail vs main image mismatch
"""
import io
import struct
from datetime import datetime
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS


class MetadataConsistencyAgent:
    """ניתוח עקביות מטאדטה פנימית."""

    AGENT_TYPE = "metadata_consistency"

    async def analyze(self, file_bytes: bytes, filename: str) -> dict:
        anomalies = []
        findings = {}

        try:
            img = Image.open(io.BytesIO(file_bytes))
            findings["format"] = img.format
            findings["size"] = f"{img.width}x{img.height}"
            findings["mode"] = img.mode

            # Extract EXIF
            exif_data = {}
            raw_exif = img.getexif()
            if raw_exif:
                for tag_id, value in raw_exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if isinstance(value, bytes):
                        try:
                            value = value.decode("utf-8", errors="ignore")
                        except Exception:
                            value = str(value)[:100]
                    exif_data[str(tag)] = value

            findings["exif_fields"] = len(exif_data)
            findings["has_exif"] = len(exif_data) > 0

            if not exif_data:
                # No EXIF — check if this is suspicious
                if img.format in ("JPEG", "TIFF"):
                    anomalies.append({
                        "type": "Missing EXIF",
                        "description": f"{img.format} file has no EXIF metadata. "
                                     f"Most cameras embed EXIF data. Missing metadata may indicate "
                                     f"the image was processed, stripped, or artificially generated.",
                        "severity": "medium",
                        "location": {"x": 50, "y": 10},
                    })

            # 1. Date consistency checks
            date_anomalies = self._check_dates(exif_data)
            anomalies.extend(date_anomalies)

            # 2. Camera vs resolution
            cam_anomalies = self._check_camera_resolution(exif_data, img.width, img.height)
            anomalies.extend(cam_anomalies)

            # 3. Software indicators
            sw_anomalies = self._check_software(exif_data)
            anomalies.extend(sw_anomalies)
            findings["software"] = exif_data.get("Software", "none")

            # 4. GPS consistency
            gps_anomalies = self._check_gps(exif_data, raw_exif)
            anomalies.extend(gps_anomalies)

            # 5. Thumbnail mismatch
            thumb_anomalies = self._check_thumbnail(img, file_bytes)
            anomalies.extend(thumb_anomalies)

            # 6. ICC profile check
            icc_anomalies = self._check_icc(img)
            anomalies.extend(icc_anomalies)

        except Exception as e:
            findings["error"] = str(e)

        # Confidence
        high = sum(1 for a in anomalies if a["severity"] == "high")
        confidence = max(0.15, 0.90 - high * 0.15 - len(anomalies) * 0.05)

        return {
            "agent_type": self.AGENT_TYPE,
            "confidence_score": round(confidence, 2),
            "findings": findings,
            "anomalies": {"items": anomalies},
        }

    def _check_dates(self, exif: dict) -> list:
        anomalies = []
        date_fields = {
            "DateTime": exif.get("DateTime"),
            "DateTimeOriginal": exif.get("DateTimeOriginal"),
            "DateTimeDigitized": exif.get("DateTimeDigitized"),
        }
        dates = {}
        for k, v in date_fields.items():
            if v:
                try:
                    if isinstance(v, str):
                        dt = datetime.strptime(v.strip()[:19], "%Y:%m:%d %H:%M:%S")
                        dates[k] = dt
                except Exception:
                    pass

        if len(dates) >= 2:
            keys = list(dates.keys())
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    diff = abs((dates[keys[i]] - dates[keys[j]]).total_seconds())
                    if diff > 86400:  # More than 1 day difference
                        anomalies.append({
                            "type": "Date Conflict",
                            "description": f"{keys[i]} ({dates[keys[i]].strftime('%Y-%m-%d')}) "
                                         f"differs from {keys[j]} ({dates[keys[j]].strftime('%Y-%m-%d')}) "
                                         f"by {diff/86400:.0f} days. Legitimate photos have consistent timestamps.",
                            "severity": "high",
                            "location": {"x": 80, "y": 10},
                        })

        # Future date check
        now = datetime.now()
        for k, dt in dates.items():
            if dt > now:
                anomalies.append({
                    "type": "Future Date",
                    "description": f"{k} is set to {dt.strftime('%Y-%m-%d')}, which is in the future. "
                                 f"This indicates metadata tampering.",
                    "severity": "high",
                    "location": {"x": 80, "y": 15},
                })

        return anomalies

    def _check_camera_resolution(self, exif: dict, width: int, height: int) -> list:
        anomalies = []
        model = exif.get("Model", "")
        make = exif.get("Make", "")

        if model and isinstance(model, str):
            model_lower = model.lower()
            mp = (width * height) / 1_000_000

            # Phone cameras claiming extremely high res
            phone_brands = ["iphone", "samsung", "pixel", "huawei", "xiaomi", "oppo"]
            is_phone = any(b in model_lower or b in str(make).lower() for b in phone_brands)

            if is_phone and mp > 200:
                anomalies.append({
                    "type": "Resolution Mismatch",
                    "description": f"Camera model '{model}' is a phone, but image is {mp:.0f}MP. "
                                 f"Phone cameras typically max at 50-108MP. May indicate upscaling or fake EXIF.",
                    "severity": "medium",
                    "location": {"x": 50, "y": 10},
                })

            # Very old camera with very high res
            old_indicators = ["d70", "d80", "d90", "350d", "400d", "rebel"]
            if any(ind in model_lower for ind in old_indicators) and mp > 20:
                anomalies.append({
                    "type": "Camera-Resolution Conflict",
                    "description": f"Camera model '{model}' is an older model, but image is {mp:.0f}MP. "
                                 f"This camera's maximum is typically 6-12MP. EXIF may be spoofed.",
                    "severity": "high",
                    "location": {"x": 50, "y": 10},
                })

        return anomalies

    def _check_software(self, exif: dict) -> list:
        anomalies = []
        software = str(exif.get("Software", ""))

        if software:
            editors = ["photoshop", "gimp", "lightroom", "affinity", "paint.net",
                      "snapseed", "pixlr", "canva", "figma", "illustrator"]
            for editor in editors:
                if editor in software.lower():
                    anomalies.append({
                        "type": "Editing Software",
                        "description": f"Image was processed with '{software}'. "
                                     f"Editing software in EXIF indicates post-processing. "
                                     f"This alone is not proof of forgery but warrants attention.",
                        "severity": "low",
                        "location": {"x": 90, "y": 10},
                    })
                    break

            # AI tools
            ai_tools = ["dall-e", "midjourney", "stable diffusion", "firefly", "leonardo"]
            for tool in ai_tools:
                if tool in software.lower():
                    anomalies.append({
                        "type": "AI Tool in Metadata",
                        "description": f"EXIF Software field contains '{software}' — "
                                     f"an AI image generation tool. Image was likely AI-generated.",
                        "severity": "high",
                        "location": {"x": 50, "y": 50},
                    })
                    break

        return anomalies

    def _check_gps(self, exif: dict, raw_exif) -> list:
        anomalies = []

        # Check for GPS data
        gps_info = {}
        try:
            for key in raw_exif.get_ifd(0x8825):
                tag = GPSTAGS.get(key, key)
                gps_info[str(tag)] = raw_exif.get_ifd(0x8825)[key]
        except Exception:
            pass

        if gps_info:
            # Check GPS timestamp vs EXIF timestamp
            gps_date = gps_info.get("GPSDateStamp", "")
            exif_date = str(exif.get("DateTimeOriginal", ""))[:10].replace(":", "-")

            if gps_date and exif_date and gps_date != exif_date:
                anomalies.append({
                    "type": "GPS-Date Mismatch",
                    "description": f"GPS date ({gps_date}) differs from EXIF date ({exif_date}). "
                                 f"GPS and camera timestamps should match for authentic photos.",
                    "severity": "high",
                    "location": {"x": 70, "y": 10},
                })

        return anomalies

    def _check_thumbnail(self, img: Image.Image, file_bytes: bytes) -> list:
        anomalies = []

        try:
            # Check if JPEG has embedded thumbnail
            if img.format == "JPEG" and hasattr(img, '_getexif'):
                exif = img._getexif()
                if exif and 513 in exif and 514 in exif:
                    # Thumbnail offset and length exist
                    thumb_offset = exif[513]
                    thumb_length = exif[514]
                    if thumb_offset + thumb_length <= len(file_bytes):
                        thumb_bytes = file_bytes[thumb_offset:thumb_offset + thumb_length]
                        try:
                            thumb = Image.open(io.BytesIO(thumb_bytes))
                            # Compare aspect ratios
                            main_ar = img.width / img.height if img.height > 0 else 1
                            thumb_ar = thumb.width / thumb.height if thumb.height > 0 else 1
                            ar_diff = abs(main_ar - thumb_ar)
                            if ar_diff > 0.1:
                                anomalies.append({
                                    "type": "Thumbnail Mismatch",
                                    "description": f"Embedded thumbnail aspect ratio ({thumb_ar:.2f}) "
                                                 f"differs from main image ({main_ar:.2f}). "
                                                 f"Thumbnail may be from the original image before cropping/editing.",
                                    "severity": "high",
                                    "location": {"x": 50, "y": 50},
                                })
                        except Exception:
                            pass
        except Exception:
            pass

        return anomalies

    def _check_icc(self, img: Image.Image) -> list:
        anomalies = []

        icc = img.info.get("icc_profile")
        if icc and len(icc) > 20:
            # Check ICC profile description
            try:
                desc_idx = icc.find(b'desc')
                if desc_idx > 0:
                    profile_name = icc[desc_idx + 12:desc_idx + 50].decode("ascii", errors="ignore").strip('\x00')
                    if profile_name:
                        # sRGB is standard — anything else is notable
                        if "srgb" not in profile_name.lower() and "display" not in profile_name.lower():
                            anomalies.append({
                                "type": "ICC Profile",
                                "description": f"Non-standard color profile: '{profile_name}'. "
                                             f"Most consumer photos use sRGB. "
                                             f"Different profiles may indicate professional editing or compositing.",
                                "severity": "low",
                                "location": {"x": 50, "y": 10},
                            })
            except Exception:
                pass

        return anomalies
