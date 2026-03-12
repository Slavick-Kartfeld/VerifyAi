"""
Orchestrator V2 — מנצח ראשי משופר.
4 שלבים: (1) סוכנים במקביל עם timeout (2) הצלבה (3) Red Team (4) פסיקה סופית.
"""
import asyncio
from app.agents.forensic_agent import ForensicTechnicalAgent
from app.agents.vision_agents import PhysicalAgent, ContextualAgent, AIGenerationAgent
from app.agents.cross_reference import CrossReferenceEngine
from app.agents.red_team_agent import RedTeamAgent
from app.agents.copy_move_agent import CopyMoveAgent
from app.agents.frequency_agent import FrequencyAnalysisAgent
from app.agents.audio_agent import AudioDeepfakeAgent
from app.agents.video_agent import VideoForensicAgent
from app.agents.document_agent import DocumentForensicAgent
from app.agents.metadata_agent import MetadataConsistencyAgent

AGENT_TIMEOUT    = 45   # seconds per agent
RED_TEAM_TIMEOUT = 60   # seconds for red team (calls Claude multiple times)


async def _run_agent_with_timeout(agent, file_bytes: bytes, filename: str, timeout: int) -> dict:
    """Run a single agent with a timeout. Returns a safe fallback on timeout/error."""
    try:
        return await asyncio.wait_for(
            agent.analyze(file_bytes, filename),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        return {
            "agent_type": getattr(agent, "AGENT_TYPE", "unknown"),
            "confidence_score": 0.5,
            "findings": {"error": f"Agent timed out after {timeout}s"},
            "anomalies": {"items": [{
                "type": "Timeout",
                "description": f"Agent did not complete within {timeout} seconds.",
                "severity": "low",
                "location": {"x": 50, "y": 50},
            }]},
        }
    except Exception as e:
        return {
            "agent_type": getattr(agent, "AGENT_TYPE", "unknown"),
            "confidence_score": 0.5,
            "findings": {"error": str(e)[:200]},
            "anomalies": {"items": []},
        }


class Orchestrator:

    def __init__(self):
        self.forensic   = ForensicTechnicalAgent()
        self.physical   = PhysicalAgent()
        self.contextual = ContextualAgent()
        self.ai_gen     = AIGenerationAgent()
        self.copy_move  = CopyMoveAgent()
        self.frequency  = FrequencyAnalysisAgent()
        self.audio      = AudioDeepfakeAgent()
        self.video      = VideoForensicAgent()
        self.document   = DocumentForensicAgent()
        self.metadata   = MetadataConsistencyAgent()
        self.cross_ref  = CrossReferenceEngine()
        self.red_team   = RedTeamAgent()

    async def analyze(self, file_bytes: bytes, filename: str, media_type: str) -> dict:
        """
        ניתוח מלא ב-4 שלבים:
        1. סוכנים מקצועיים במקביל — כל אחד עם timeout
        2. הצלבה ראשונית
        3. Red Team מאתגר
        4. פסיקה סופית (מתוקנת לאור Red Team)
        """

        # ── שלב 1: סוכנים במקביל עם timeout לכל אחד ──
        agents = self._select_agents(media_type)
        tasks  = [
            _run_agent_with_timeout(agent, file_bytes, filename, AGENT_TIMEOUT)
            for agent in agents
        ]
        raw_results  = await asyncio.gather(*tasks)
        agent_results = [r for r in raw_results if isinstance(r, dict)]

        # ── שלב 2: הצלבה ראשונית ──
        initial_cross = self.cross_ref.analyze(agent_results)

        # ── שלב 3: Red Team (עם timeout משלו) ──
        try:
            red_team_result = await asyncio.wait_for(
                self.red_team.challenge(file_bytes, filename, agent_results, initial_cross),
                timeout=RED_TEAM_TIMEOUT
            )
        except asyncio.TimeoutError:
            red_team_result = {
                "summary": "Red Team analysis timed out.",
                "threat_level": "low",
                "challenges": [],
                "blind_spots": [],
                "recommendations": ["Consider re-running analysis for full Red Team evaluation."],
                "confidence_adjustment": 0,
            }
        except Exception as e:
            red_team_result = {
                "summary": f"Red Team error: {str(e)[:100]}",
                "threat_level": "low",
                "challenges": [],
                "blind_spots": [],
                "recommendations": [],
                "confidence_adjustment": 0,
            }

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
            return [self.forensic, self.physical, self.contextual, self.ai_gen,
                    self.copy_move, self.frequency, self.metadata]
        elif media_type == "video":
            return [self.physical, self.contextual, self.video, self.frequency]
        elif media_type == "audio":
            return [self.audio]                              # audio only — forensic is image-only
        elif media_type == "document":
            return [self.contextual, self.document, self.metadata]
        return [self.forensic]

    def _final_verdict(self, cross_ref: dict, red_team: dict) -> dict:
        """פסיקה סופית — מתחשבת בביקורת הצוות האדום."""
        base_score   = cross_ref.get("combined_score", 0.5)
        base_verdict = cross_ref.get("final_verdict", "inconclusive")

        adj            = red_team.get("confidence_adjustment", 0)
        adjusted_score = max(0.05, min(0.99, base_score + adj))

        threat = red_team.get("threat_level", "low")
        high_challenges = [
            c for c in red_team.get("challenges", [])
            if c.get("type") == "verdict_challenge"
        ]

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
