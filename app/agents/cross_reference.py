"""
מנוע הצלבה (Cross-Reference Engine)
מצליב ממצאי כל הסוכנים, מחשב ציון ביטחון משוקלל ומפיק פסיקה.
"""


class CrossReferenceEngine:

    # משקלות לכל סוכן
    WEIGHTS = {
        "forensic_technical": 0.35,
        "physical": 0.25,
        "contextual": 0.20,
        "ai_generation": 0.20,
    }

    CONFIDENCE_THRESHOLD = 0.75  # מעל = אותנטי, מתחת = inconclusive/forged

    def analyze(self, agent_results: list[dict]) -> dict:
        if not agent_results:
            return {
                "combined_score": 0.5,
                "final_verdict": "inconclusive",
                "reasoning": "לא התקבלו ממצאים מסוכנים."
            }

        # חישוב ציון משוקלל
        total_weight = 0
        weighted_sum = 0

        for result in agent_results:
            agent_type = result.get("agent_type", "")
            weight = self.WEIGHTS.get(agent_type, 0.1)
            score = result.get("confidence_score", 0.5)
            weighted_sum += score * weight
            total_weight += weight

        combined_score = round(weighted_sum / total_weight, 3) if total_weight > 0 else 0.5

        # ספירת אנומליות לפי חומרה
        all_anomalies = []
        for result in agent_results:
            items = result.get("anomalies", {}).get("items", [])
            all_anomalies.extend(items)

        high_count = sum(1 for a in all_anomalies if a.get("severity") == "high")
        medium_count = sum(1 for a in all_anomalies if a.get("severity") == "medium")
        total_count = len(all_anomalies)

        # קביעת פסיקה
        if high_count >= 3 or (high_count >= 2 and combined_score < 0.5):
            verdict = "forged"
        elif combined_score >= self.CONFIDENCE_THRESHOLD and high_count == 0:
            verdict = "authentic"
        else:
            verdict = "inconclusive"

        # בניית נימוק
        reasoning_parts = []
        reasoning_parts.append(
            f"הופעלו {len(agent_results)} סוכני ניתוח. "
            f"ציון ביטחון משוקלל: {combined_score:.1%}."
        )
        reasoning_parts.append(
            f"זוהו {total_count} אנומליות: "
            f"{high_count} בחומרה גבוהה, {medium_count} בחומרה בינונית."
        )

        # בדיקת הסכמה בין סוכנים
        low_confidence_agents = [
            r["agent_type"] for r in agent_results
            if r.get("confidence_score", 1) < 0.6
        ]
        if low_confidence_agents:
            agent_names = {
                "forensic_technical": "פורנזי-טכני",
                "physical": "פיזיקלי",
                "contextual": "הקשרי",
                "ai_generation": "זיהוי AI"
            }
            names = [agent_names.get(a, a) for a in low_confidence_agents]
            reasoning_parts.append(
                f"ציון נמוך בסוכנים: {', '.join(names)}."
            )

        # בדיקת AI generation
        for result in agent_results:
            if result.get("agent_type") == "ai_generation":
                findings = result.get("findings", {})
                if findings.get("is_ai_generated"):
                    tool = findings.get("likely_tool", "לא ידוע")
                    reasoning_parts.append(f"התמונה זוהתה כנוצרת על ידי AI (כלי: {tool}).")

        if verdict == "inconclusive":
            reasoning_parts.append("מומלץ לשלב מומחה אנושי (HITL) לאימות הממצאים.")

        reasoning = " ".join(reasoning_parts)

        return {
            "combined_score": combined_score,
            "final_verdict": verdict,
            "reasoning": reasoning,
            "anomaly_summary": {
                "total": total_count,
                "high": high_count,
                "medium": medium_count,
                "low": total_count - high_count - medium_count
            }
        }
