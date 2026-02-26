"""
Red Team Agent — סוכן אדום מאתגר.
מקבל ממצאי כל הסוכנים, מנסה למצוא נקודות עיוורות,
false positives/negatives, ומפיק המלצות שיפור.
"""
import io
import hashlib
from datetime import datetime
from PIL import Image
import numpy as np


class RedTeamAgent:
    """צוות אדום — מאתגר את הממצאים של שאר הסוכנים."""

    AGENT_TYPE = "red_team"

    # לוג למידה מצטבר (in-memory, בעתיד — DB)
    learning_log: list = []

    async def challenge(
        self,
        file_bytes: bytes,
        filename: str,
        agent_results: list[dict],
        cross_reference: dict,
    ) -> dict:
        """מאתגר את ממצאי המערכת ומחזיר ביקורת + המלצות."""

        challenges = []
        recommendations = []
        blind_spots = []
        confidence_adjustments = []

        # 1. בדיקת False Positives — האם אנומליות ELA לגיטימיות?
        fp_checks = self._check_false_positives(agent_results, file_bytes)
        challenges.extend(fp_checks["challenges"])
        if fp_checks["adjustment"]:
            confidence_adjustments.append(fp_checks["adjustment"])

        # 2. בדיקת False Negatives — מה הסוכנים פיספסו?
        fn_checks = self._check_false_negatives(agent_results, file_bytes)
        blind_spots.extend(fn_checks["blind_spots"])

        # 3. בדיקת עקביות בין סוכנים
        consistency = self._check_cross_consistency(agent_results)
        challenges.extend(consistency["challenges"])
        if consistency["adjustment"]:
            confidence_adjustments.append(consistency["adjustment"])

        # 4. בדיקת edge cases
        edge = self._check_edge_cases(file_bytes, agent_results)
        challenges.extend(edge["challenges"])
        recommendations.extend(edge["recommendations"])

        # 5. ניתוח meta — האם הפסיקה הכללית הגיונית?
        meta = self._challenge_verdict(cross_reference, agent_results)
        challenges.extend(meta["challenges"])

        # חישוב התאמת ביטחון
        adj = 0
        if confidence_adjustments:
            adj = sum(confidence_adjustments) / len(confidence_adjustments)

        # הפקת המלצות שיפור
        recommendations.extend(self._generate_recommendations(
            challenges, blind_spots, agent_results
        ))

        # תיעוד בלוג למידה
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "file_hash": hashlib.sha256(file_bytes).hexdigest()[:16],
            "challenges_count": len(challenges),
            "blind_spots_count": len(blind_spots),
            "confidence_adjustment": round(adj, 3),
            "verdict_challenged": len(meta["challenges"]) > 0,
        }
        self.learning_log.append(entry)

        # חישוב threat level
        threat = self._calc_threat_level(challenges, blind_spots)

        return {
            "agent_type": self.AGENT_TYPE,
            "challenges": challenges,
            "blind_spots": blind_spots,
            "recommendations": recommendations,
            "confidence_adjustment": round(adj, 3),
            "threat_level": threat,
            "learning_log_size": len(self.learning_log),
            "summary": self._build_summary(challenges, blind_spots, recommendations, threat),
        }

    # ─── CHALLENGE MODULES ───

    def _check_false_positives(self, results: list, file_bytes: bytes) -> dict:
        """בדיקה: האם אנומליות ELA נובעות מדחיסה לגיטימית?"""
        challenges = []
        adjustment = None

        forensic = next((r for r in results if r["agent_type"] == "forensic_technical"), None)
        if not forensic:
            return {"challenges": [], "adjustment": None}

        anomalies = forensic.get("anomalies", {})
        items = anomalies.get("items", []) if isinstance(anomalies, dict) else anomalies

        ela_anomalies = [a for a in items if "ELA" in a.get("type", "")]
        if ela_anomalies:
            # בדיקה: האם התמונה עברה דחיסה כפולה לגיטימית?
            try:
                img = Image.open(io.BytesIO(file_bytes))
                if img.format == "JPEG":
                    # בדיקת מספר שמירות — תמונות מוואטסאפ/טלגרם נדחסות פעמיים
                    pixels = img.width * img.height
                    bpp = len(file_bytes) / pixels if pixels > 0 else 0

                    if bpp < 0.3:  # דחיסה גבוהה — סביר שזה WhatsApp/social
                        challenges.append({
                            "type": "false_positive_risk",
                            "target_agent": "forensic_technical",
                            "target_anomaly": "ELA",
                            "challenge": "אנומליית ELA עשויה להיות תוצאה של דחיסה כפולה לגיטימית "
                                        "(שיתוף ברשתות חברתיות/WhatsApp). יחס דחיסה מצביע על שמירות מרובות.",
                            "severity": "medium",
                            "impact": "ציון הפורנזי עשוי להיות נמוך מדי (false positive).",
                        })
                        adjustment = 0.05  # הגדלת ביטחון קלה
            except Exception:
                pass

        # בדיקת מטאדטה — האם חוסר EXIF באמת חשוד?
        meta_anomalies = [a for a in items if "מטאדטה" in a.get("type", "")]
        if meta_anomalies:
            challenges.append({
                "type": "false_positive_risk",
                "target_agent": "forensic_technical",
                "target_anomaly": "metadata",
                "challenge": "חוסר מטאדטה EXIF אינו בהכרח מעיד על זיוף. "
                            "פלטפורמות רבות (Twitter, Facebook, WhatsApp) מסירות EXIF באופן אוטומטי.",
                "severity": "low",
                "impact": "חוסר EXIF לבדו אינו ראיה מספקת.",
            })

        return {"challenges": challenges, "adjustment": adjustment}

    def _check_false_negatives(self, results: list, file_bytes: bytes) -> dict:
        """מה הסוכנים פיספסו?"""
        blind_spots = []

        # בדיקה: האם יש סוכנים שלא מצאו כלום?
        for r in results:
            items = r.get("anomalies", {})
            if isinstance(items, dict):
                items = items.get("items", [])
            if not items and r.get("confidence_score", 0) > 0.8:
                blind_spots.append({
                    "agent": r["agent_type"],
                    "issue": f"סוכן {r['agent_type']} דיווח ביטחון גבוה ({r['confidence_score']:.0%}) "
                            f"ללא ממצאים. האם הוא לא בדק מספיק?",
                    "risk": "false_negative",
                })

        # בדיקה: האם בדקנו clone/copy-move?
        agent_types = [r["agent_type"] for r in results]
        all_anomaly_types = []
        for r in results:
            items = r.get("anomalies", {})
            if isinstance(items, dict):
                items = items.get("items", [])
            all_anomaly_types.extend([a.get("type", "") for a in items])

        if "clone_detection" not in " ".join(all_anomaly_types).lower():
            blind_spots.append({
                "agent": "system",
                "issue": "לא בוצע ניתוח copy-move/clone detection. "
                        "טכניקת זיוף נפוצה שעלולה לחמוק מהסוכנים הנוכחיים.",
                "risk": "missing_capability",
            })

        # בדיקה: האם בדקנו GAN artifacts?
        if "ai_generation" not in agent_types:
            blind_spots.append({
                "agent": "system",
                "issue": "סוכן זיהוי AI לא הופעל. תמונות GAN/diffusion עלולות לעבור ללא זיהוי.",
                "risk": "missing_agent",
            })

        return {"blind_spots": blind_spots}

    def _check_cross_consistency(self, results: list) -> dict:
        """בדיקת עקביות בין ממצאי סוכנים שונים."""
        challenges = []
        adjustment = None

        scores = {r["agent_type"]: r.get("confidence_score", 0.5) for r in results}

        # בדיקת פער גדול בציונים
        if len(scores) >= 2:
            max_s = max(scores.values())
            min_s = min(scores.values())
            gap = max_s - min_s

            if gap > 0.3:
                high_agent = [k for k, v in scores.items() if v == max_s][0]
                low_agent = [k for k, v in scores.items() if v == min_s][0]
                challenges.append({
                    "type": "consistency_gap",
                    "challenge": f"פער משמעותי ({gap:.0%}) בין {high_agent} ({max_s:.0%}) "
                                f"ל-{low_agent} ({min_s:.0%}). אחד מהם עלול לטעות.",
                    "severity": "high",
                    "impact": "הציון המשוקלל עשוי להיות מטעה.",
                })
                adjustment = -0.03  # הורדת ביטחון כשיש חוסר הסכמה

        # בדיקה: פורנזי אומר נקי אבל הקשרי מוצא בעיות (או להיפך)
        forensic_score = scores.get("forensic_technical", 0.5)
        contextual_score = scores.get("contextual", 0.5)
        if abs(forensic_score - contextual_score) > 0.25:
            challenges.append({
                "type": "cross_disagreement",
                "challenge": "הפורנזי וההקשרי חלוקים. ייתכן זיוף מתוחכם שעבר רק שכבה אחת, "
                            "או false positive בשכבה אחרת.",
                "severity": "medium",
                "impact": "נדרשת בחינה ידנית של נקודות החפיפה.",
            })

        return {"challenges": challenges, "adjustment": adjustment}

    def _check_edge_cases(self, file_bytes: bytes, results: list) -> dict:
        """בדיקת מקרי קצה."""
        challenges = []
        recommendations = []

        try:
            img = Image.open(io.BytesIO(file_bytes))

            # תמונה קטנה מדי לניתוח אמין
            if img.width < 200 or img.height < 200:
                challenges.append({
                    "type": "low_resolution",
                    "challenge": f"רזולוציית התמונה נמוכה ({img.width}x{img.height}). "
                                f"ניתוח ELA ו-artifacts פחות אמין בתמונות קטנות.",
                    "severity": "high",
                    "impact": "כל הממצאים צריכים להיחשב בזהירות.",
                })

            # תמונה גדולה מאוד — ייתכן שנחתכה מתמונה גדולה יותר
            if img.width > 5000 or img.height > 5000:
                recommendations.append(
                    "תמונה ברזולוציה גבוהה מאוד — שקול לבדוק חיתוך/קרופ מתמונה מקורית גדולה יותר."
                )

            # Grayscale — חלק מהבדיקות פחות רלוונטיות
            if img.mode in ("L", "LA"):
                challenges.append({
                    "type": "grayscale",
                    "challenge": "תמונה בגווני אפור. ניתוח צבע, תאורה וחלק מבדיקות AI פחות אמינים.",
                    "severity": "medium",
                    "impact": "סוכנים פיזיקלי והקשרי עשויים להחמיץ אנומליות.",
                })

        except Exception:
            challenges.append({
                "type": "parse_error",
                "challenge": "לא ניתן לפרסר את הקובץ לניתוח edge cases.",
                "severity": "low",
                "impact": "בדיקות edge cases לא בוצעו.",
            })

        return {"challenges": challenges, "recommendations": recommendations}

    def _challenge_verdict(self, cross_ref: dict, results: list) -> dict:
        """מאתגר את הפסיקה הכללית."""
        challenges = []
        verdict = cross_ref.get("final_verdict", "")
        score = cross_ref.get("combined_score", 0.5)

        # פסיקת "authentic" עם אנומליות גבוהות
        if verdict == "authentic":
            total_high = sum(
                1 for r in results
                for a in (r.get("anomalies", {}).get("items", [])
                         if isinstance(r.get("anomalies"), dict)
                         else r.get("anomalies", []))
                if a.get("severity") == "high"
            )
            if total_high > 0:
                challenges.append({
                    "type": "verdict_challenge",
                    "challenge": f"פסיקת 'authentic' למרות {total_high} אנומליות בחומרה גבוהה. "
                                f"האם המשקלות שגויים?",
                    "severity": "high",
                    "impact": "עלול להיות false negative מסוכן.",
                })

        # פסיקת "forged" עם ציון גבוה
        if verdict == "forged" and score > 0.7:
            challenges.append({
                "type": "verdict_challenge",
                "challenge": f"פסיקת 'forged' אבל ציון הביטחון גבוה ({score:.0%}). "
                            f"ייתכן שהמודל שגוי — ציון גבוה אמור להצביע על אותנטיות.",
                "severity": "medium",
                "impact": "ייתכן שהתגלתה באג בלוגיקת הפסיקה.",
            })

        return {"challenges": challenges}

    # ─── RECOMMENDATIONS ───

    def _generate_recommendations(self, challenges, blind_spots, results) -> list:
        recs = []

        high_challenges = [c for c in challenges if c.get("severity") == "high"]
        if high_challenges:
            recs.append("נמצאו אתגרים ברמה גבוהה — מומלץ לשלב מומחה HITL לאימות.")

        if len(blind_spots) >= 2:
            recs.append("זוהו מספר נקודות עיוורות. שקול להרחיב את סט הסוכנים (clone detection, frequency analysis).")

        # בדיקת מגמות מלוג הלמידה
        if len(self.learning_log) >= 5:
            recent = self.learning_log[-5:]
            avg_challenges = sum(e["challenges_count"] for e in recent) / 5
            if avg_challenges > 3:
                recs.append(f"ב-5 הניתוחים האחרונים נמצאו בממוצע {avg_challenges:.1f} אתגרים — "
                           f"המערכת עשויה להיות רגישה מדי (over-sensitive).")

        return recs

    # ─── HELPERS ───

    def _calc_threat_level(self, challenges, blind_spots) -> str:
        high = sum(1 for c in challenges if c.get("severity") == "high")
        total = len(challenges) + len(blind_spots)
        if high >= 2 or total >= 5:
            return "high"
        if high >= 1 or total >= 3:
            return "medium"
        return "low"

    def _build_summary(self, challenges, blind_spots, recs, threat) -> str:
        parts = [f"צוות אדום זיהה {len(challenges)} אתגרים ו-{len(blind_spots)} נקודות עיוורות."]
        parts.append(f"רמת איום: {threat}.")
        if recs:
            parts.append(f"הומלצו {len(recs)} שיפורים.")
        return " ".join(parts)
