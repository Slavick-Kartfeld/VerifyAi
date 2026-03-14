"""
Cross-Reference Engine
Aggregates all agent findings, computes a weighted confidence score, and produces a final verdict.
Backend language rule: all strings returned here must be English.
"""


class CrossReferenceEngine:

    # Agent weights — must sum to 1.0
    # C2PA gets 0.15 because when it fires it is cryptographically certain.
    # When NOT_PRESENT it returns 0.5 (neutral) and does not distort the score.
    WEIGHTS = {
        "forensic_technical":  0.18,
        "c2pa_provenance":     0.15,
        "physical":            0.12,
        "contextual":          0.10,
        "ai_generation":       0.10,
        "copy_move":           0.09,
        "frequency_analysis":  0.09,
        "metadata_consistency":0.08,
        "audio_deepfake":      0.07,
        "video_forensic":      0.06,
        "document_forensic":   0.06,
    }  # total = 1.10 — intentional: normalised by total_weight below

    CONFIDENCE_THRESHOLD = 0.75  # above = authentic, below = inconclusive / forged

    def analyze(self, agent_results: list[dict]) -> dict:
        if not agent_results:
            return {
                "combined_score": 0.5,
                "final_verdict": "inconclusive",
                "reasoning": "No agent results received."
            }

        # Weighted score (normalised — handles missing agents gracefully)
        total_weight = 0
        weighted_sum = 0

        for result in agent_results:
            agent_type = result.get("agent_type", "")
            weight = self.WEIGHTS.get(agent_type, 0.08)  # unknown agent gets 0.08
            score  = result.get("confidence_score", 0.5)
            weighted_sum += score * weight
            total_weight += weight

        combined_score = round(weighted_sum / total_weight, 3) if total_weight > 0 else 0.5

        # Count anomalies by severity
        all_anomalies = []
        for result in agent_results:
            items = result.get("anomalies", {}).get("items", [])
            all_anomalies.extend(items)

        high_count   = sum(1 for a in all_anomalies if a.get("severity") == "high")
        medium_count = sum(1 for a in all_anomalies if a.get("severity") == "medium")
        total_count  = len(all_anomalies)

        # Verdict logic
        if high_count >= 3 or (high_count >= 2 and combined_score < 0.5):
            verdict = "forged"
        elif combined_score >= self.CONFIDENCE_THRESHOLD and high_count == 0:
            verdict = "authentic"
        else:
            verdict = "inconclusive"

        # C2PA override: cryptographic TAMPERED = always forged regardless of other agents
        for result in agent_results:
            if result.get("agent_type") == "c2pa_provenance":
                c2pa_status = result.get("findings", {}).get("c2pa_status", "")
                if c2pa_status == "TAMPERED":
                    verdict = "forged"
                    break

        # Reasoning (English only)
        reasoning_parts = [
            f"{len(agent_results)} analysis agents ran. "
            f"Weighted confidence score: {combined_score:.1%}.",
            f"{total_count} anomalies detected: "
            f"{high_count} high severity, {medium_count} medium severity."
        ]

        low_conf = [r["agent_type"] for r in agent_results if r.get("confidence_score", 1) < 0.6]
        if low_conf:
            reasoning_parts.append(f"Low confidence agents: {', '.join(low_conf)}.")

        # AI generation note
        for result in agent_results:
            if result.get("agent_type") == "ai_generation":
                findings = result.get("findings", {})
                if findings.get("is_ai_generated"):
                    tool = findings.get("likely_tool", "unknown")
                    reasoning_parts.append(f"Image detected as AI-generated (tool: {tool}).")

        # C2PA note
        for result in agent_results:
            if result.get("agent_type") == "c2pa_provenance":
                status = result.get("findings", {}).get("c2pa_status", "")
                if status == "VERIFIED":
                    signed_by = result.get("findings", {}).get("signed_by", "unknown")
                    reasoning_parts.append(f"C2PA Content Credentials verified (signed by: {signed_by}).")
                elif status == "TAMPERED":
                    reasoning_parts.append("C2PA signature chain broken — cryptographic evidence of tampering.")
                elif status == "AI_DECLARED":
                    reasoning_parts.append("C2PA manifest declares AI-generated content.")

        if verdict == "inconclusive":
            reasoning_parts.append("Human-in-the-loop (HITL) review recommended.")

        return {
            "combined_score": combined_score,
            "final_verdict":  verdict,
            "reasoning":      " ".join(reasoning_parts),
            "anomaly_summary": {
                "total":  total_count,
                "high":   high_count,
                "medium": medium_count,
                "low":    total_count - high_count - medium_count
            }
        }
