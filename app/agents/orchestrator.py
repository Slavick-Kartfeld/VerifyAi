"""
Orchestrator V2 — מנצח ראשי משופר.
4 שלבים: (1) סוכנים במקביל (2) הצלבה (3) Red Team (4) פסיקה סופית.
"""
import asyncio
from app.agents.forensic_agent import ForensicTechnicalAgent
from app.agents.vision_agents import PhysicalAgent, ContextualAgent, AIGenerationAgent
from app.agents.cross_reference import CrossReferenceEngine
from app.agents.red_team_agent import RedTeamAgent


class Orchestrator:

    def __init__(self):
        self.forensic = ForensicTechnicalAgent()
        self.physical = PhysicalAgent()
        self.contextual = ContextualAgent()
        self.ai_gen = AIGenerationAgent()
        self.cross_ref = CrossReferenceEngine()
        self.red_team = RedTeamAgent()

    async def analyze(self, file_bytes: bytes, filename: str, media_type: str) -> dict:
        """
        ניתוח מלא ב-4 שלבים:
        1. סוכנים מקצועיים במקביל
        2. הצלבה ראשונית
        3. Red Team מאתגר
        4. פסיקה סופית (מתוקנת לאור Red Team)
        """

        # ── שלב 1: סוכנים במקביל ──
        agents = self._select_agents(media_type)
        tasks = [agent.analyze(file_bytes, filename) for agent in agents]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        agent_results = [r for r in raw_results if isinstance(r, dict)]

        # ── שלב 2: הצלבה ראשונית ──
        initial_cross = self.cross_ref.analyze(agent_results)

        # ── שלב 3: Red Team ──
        red_team_result = await self.red_team.challenge(
            file_bytes, filename, agent_results, initial_cross
        )

        # ── שלב 4: פסיקה סופית מתוקנת ──
        final = self._final_verdict(initial_cross, red_team_result)

        return {
            "agent_results": agent_results,
            "cross_reference": initial_cross,
            "red_team": red_team_result,
            "verdict": final["verdict"],
            "confidence_score": final["confidence"],
            "hitl_required": final["hitl_required"],
        }

    def _select_agents(self, media_type: str) -> list:
        if media_type == "image":
            return [self.forensic, self.physical, self.contextual, self.ai_gen]
        elif media_type == "video":
            return [self.forensic, self.physical, self.contextual]
        elif media_type == "audio":
            return [self.forensic]
        elif media_type == "document":
            return [self.forensic, self.contextual]
        return [self.forensic]

    def _final_verdict(self, cross_ref: dict, red_team: dict) -> dict:
        """פסיקה סופית — מתחשבת בביקורת הצוות האדום."""
        base_score = cross_ref.get("combined_score", 0.5)
        base_verdict = cross_ref.get("final_verdict", "inconclusive")

        # התאמת ביטחון לפי Red Team
        adj = red_team.get("confidence_adjustment", 0)
        adjusted_score = max(0.05, min(0.99, base_score + adj))

        # אם Red Team מזהה threat level גבוה — דרוש HITL
        threat = red_team.get("threat_level", "low")
        high_challenges = [
            c for c in red_team.get("challenges", [])
            if c.get("type") == "verdict_challenge"
        ]

        # אם ה-Red Team מאתגר את הפסיקה ברמה גבוהה — שנה ל-inconclusive
        if high_challenges and base_verdict in ("authentic", "forged"):
            adjusted_verdict = "inconclusive"
        else:
            if adjusted_score >= 0.75 and base_verdict == "authentic":
                adjusted_verdict = "authentic"
            elif base_verdict == "forged":
                adjusted_verdict = "forged"
            else:
                adjusted_verdict = "inconclusive"

        hitl = (
            adjusted_verdict == "inconclusive"
            or threat in ("high", "medium")
            or len(red_team.get("blind_spots", [])) >= 2
        )

        return {
            "verdict": adjusted_verdict,
            "confidence": round(adjusted_score, 3),
            "hitl_required": hitl,
        }
