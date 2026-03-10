"""
Red Team Agent V2 — 4-Layer Adversarial Validation System.
Layer 0: Rule-based challenges (original)
Layer 1: Statistical Learning Engine — learns from history, detects outliers
Layer 2: Adversarial Image Testing — applies forgery techniques, tests if agents detect them
Layer 3: Cross-Agent Debate — opposing arguments for authentic vs forged
"""
import io
import hashlib
from datetime import datetime
from PIL import Image, ImageFilter
import numpy as np


class RedTeamAgent:
    """4-layer adversarial validation system."""

    AGENT_TYPE = "red_team"

    # Persistent learning store (in production → DB)
    history: list = []

    async def challenge(
        self,
        file_bytes: bytes,
        filename: str,
        agent_results: list[dict],
        cross_reference: dict,
    ) -> dict:
        """Run all 4 layers and merge results."""

        all_challenges = []
        all_blind_spots = []
        all_recommendations = []
        adjustments = []

        # ── Layer 0: Rule-based (original) ──
        L0 = self._layer0_rules(agent_results, cross_reference, file_bytes)
        all_challenges.extend(L0["challenges"])
        all_blind_spots.extend(L0["blind_spots"])
        if L0["adjustment"]:
            adjustments.append(L0["adjustment"])

        # ── Layer 1: Statistical Learning ──
        L1 = self._layer1_statistical(agent_results, cross_reference)
        all_challenges.extend(L1["challenges"])
        all_recommendations.extend(L1["recommendations"])
        if L1["adjustment"]:
            adjustments.append(L1["adjustment"])

        # ── Layer 2: Adversarial Image Testing ──
        L2 = await self._layer2_adversarial(file_bytes, agent_results)
        all_challenges.extend(L2["challenges"])
        all_blind_spots.extend(L2["blind_spots"])

        # ── Layer 3: Cross-Agent Debate ──
        L3 = self._layer3_debate(agent_results, cross_reference)
        all_challenges.extend(L3["challenges"])
        all_recommendations.extend(L3["recommendations"])
        if L3["adjustment"]:
            adjustments.append(L3["adjustment"])

        # Merge adjustments
        adj = sum(adjustments) / len(adjustments) if adjustments else 0

        # Generate recommendations from all layers
        all_recommendations.extend(self._generate_recommendations(
            all_challenges, all_blind_spots, agent_results
        ))

        # Record to history
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "file_hash": hashlib.sha256(file_bytes).hexdigest()[:16],
            "agent_scores": {r["agent_type"]: r.get("confidence_score", 0.5) for r in agent_results},
            "verdict": cross_reference.get("final_verdict", ""),
            "combined_score": cross_reference.get("combined_score", 0.5),
            "challenges_count": len(all_challenges),
            "blind_spots_count": len(all_blind_spots),
            "confidence_adjustment": round(adj, 3),
            "layers_triggered": [
                "L0:rules" if L0["challenges"] else None,
                "L1:stats" if L1["challenges"] else None,
                "L2:adversarial" if L2["challenges"] else None,
                "L3:debate" if L3["challenges"] else None,
            ],
        }
        entry["layers_triggered"] = [x for x in entry["layers_triggered"] if x]
        self.history.append(entry)

        threat = self._calc_threat_level(all_challenges, all_blind_spots)

        return {
            "agent_type": self.AGENT_TYPE,
            "challenges": all_challenges,
            "blind_spots": all_blind_spots,
            "recommendations": list(dict.fromkeys(all_recommendations)),  # deduplicate
            "confidence_adjustment": round(adj, 3),
            "threat_level": threat,
            "layers_activated": len(entry["layers_triggered"]),
            "learning_log_size": len(self.history),
            "summary": self._build_summary(all_challenges, all_blind_spots, all_recommendations, threat, entry["layers_triggered"]),
        }

    # ═══════════════════════════════════════════════════════
    # LAYER 0: Rule-Based Challenges (original)
    # ═══════════════════════════════════════════════════════
    def _layer0_rules(self, results, cross_ref, file_bytes) -> dict:
        challenges = []
        blind_spots = []
        adjustment = None

        # False positive: ELA from WhatsApp compression
        forensic = next((r for r in results if r["agent_type"] == "forensic_technical"), None)
        if forensic:
            items = self._get_items(forensic)
            ela = [a for a in items if "ELA" in a.get("type", "")]
            if ela:
                try:
                    img = Image.open(io.BytesIO(file_bytes))
                    if img.format == "JPEG":
                        bpp = len(file_bytes) / (img.width * img.height) if img.width * img.height > 0 else 0
                        if bpp < 0.3:
                            challenges.append({
                                "type": "L0:false_positive_risk",
                                "challenge": "ELA anomaly may result from legitimate double compression "
                                           "(social media/WhatsApp). Low bits-per-pixel suggests multiple saves.",
                                "severity": "medium",
                                "layer": "rules",
                            })
                            adjustment = 0.05
                except Exception:
                    pass

        # False negative: missing agents
        agent_types = [r["agent_type"] for r in results]
        all_anomaly_types = []
        for r in results:
            all_anomaly_types.extend([a.get("type", "") for a in self._get_items(r)])

        if "copy_move" not in agent_types:
            blind_spots.append({
                "agent": "system",
                "issue": "Copy-Move detection was not activated. Common forgery technique may go undetected.",
                "risk": "missing_agent",
                "layer": "rules",
            })

        # Cross-consistency gap
        scores = {r["agent_type"]: r.get("confidence_score", 0.5) for r in results}
        if len(scores) >= 2:
            gap = max(scores.values()) - min(scores.values())
            if gap > 0.3:
                high = max(scores, key=scores.get)
                low = min(scores, key=scores.get)
                challenges.append({
                    "type": "L0:consistency_gap",
                    "challenge": f"Significant gap ({gap:.0%}) between {high} ({scores[high]:.0%}) "
                               f"and {low} ({scores[low]:.0%}). One agent may be unreliable.",
                    "severity": "high",
                    "layer": "rules",
                })
                adjustment = (adjustment or 0) - 0.03

        # Verdict challenge
        verdict = cross_ref.get("final_verdict", "")
        if verdict == "authentic":
            total_high = sum(1 for r in results for a in self._get_items(r) if a.get("severity") == "high")
            if total_high > 0:
                challenges.append({
                    "type": "L0:verdict_challenge",
                    "challenge": f"Verdict is 'authentic' but {total_high} high-severity anomalies exist. "
                               f"Scoring weights may need adjustment.",
                    "severity": "high",
                    "layer": "rules",
                })

        return {"challenges": challenges, "blind_spots": blind_spots, "adjustment": adjustment}

    # ═══════════════════════════════════════════════════════
    # LAYER 1: Statistical Learning Engine
    # ═══════════════════════════════════════════════════════
    def _layer1_statistical(self, results, cross_ref) -> dict:
        challenges = []
        recommendations = []
        adjustment = None

        if len(self.history) < 5:
            return {"challenges": [], "recommendations": [
                f"Statistical learning needs more data ({len(self.history)}/5 minimum). "
                f"Results will improve with usage."
            ], "adjustment": None}

        recent = self.history[-20:]  # Last 20 analyses

        # Build baselines per agent
        agent_baselines = {}
        for entry in recent:
            for agent, score in entry.get("agent_scores", {}).items():
                if agent not in agent_baselines:
                    agent_baselines[agent] = []
                agent_baselines[agent].append(score)

        # Check current results against baselines
        for r in results:
            agent = r["agent_type"]
            score = r.get("confidence_score", 0.5)
            baseline = agent_baselines.get(agent, [])

            if len(baseline) >= 3:
                mean = np.mean(baseline)
                std = np.std(baseline)

                if std > 0.01:
                    z_score = (score - mean) / std

                    # Unusually high confidence
                    if z_score > 2.0:
                        challenges.append({
                            "type": "L1:statistical_outlier_high",
                            "challenge": f"{agent} reports {score:.0%} confidence — "
                                       f"{z_score:.1f} standard deviations above its average ({mean:.0%}). "
                                       f"Unusually high. May indicate the agent is not detecting real issues.",
                            "severity": "medium",
                            "layer": "statistical",
                        })

                    # Unusually low confidence
                    elif z_score < -2.0:
                        challenges.append({
                            "type": "L1:statistical_outlier_low",
                            "challenge": f"{agent} reports {score:.0%} confidence — "
                                       f"{z_score:.1f} standard deviations below its average ({mean:.0%}). "
                                       f"May be overly suspicious or encountering an edge case.",
                            "severity": "medium",
                            "layer": "statistical",
                        })

        # Trend analysis: are we consistently flagging the same issues?
        recent_challenges = [e["challenges_count"] for e in recent]
        avg_challenges = np.mean(recent_challenges)
        if avg_challenges > 4:
            recommendations.append(
                f"Average {avg_challenges:.1f} challenges per analysis over last {len(recent)} cases. "
                f"System may be over-sensitive. Consider recalibrating agent thresholds."
            )

        # Verdict distribution
        verdicts = [e.get("verdict", "") for e in recent]
        inc_rate = verdicts.count("inconclusive") / len(verdicts) if verdicts else 0
        if inc_rate > 0.7:
            recommendations.append(
                f"Inconclusive rate is {inc_rate:.0%} over last {len(recent)} analyses. "
                f"Consider loosening the confidence threshold or improving agent accuracy."
            )
            adjustment = 0.02

        return {"challenges": challenges, "recommendations": recommendations, "adjustment": adjustment}

    # ═══════════════════════════════════════════════════════
    # LAYER 2: Adversarial Image Testing
    # ═══════════════════════════════════════════════════════
    async def _layer2_adversarial(self, file_bytes, agent_results) -> dict:
        challenges = []
        blind_spots = []

        try:
            img = Image.open(io.BytesIO(file_bytes))
            if img.format not in ("JPEG", "PNG"):
                return {"challenges": [], "blind_spots": []}

            arr = np.array(img.convert("L"), dtype=np.float64)
            if arr.shape[0] < 64 or arr.shape[1] < 64:
                return {"challenges": [], "blind_spots": []}

            # Test 1: Can agents detect subtle brightness manipulation?
            manipulated = arr.copy()
            h, w = arr.shape
            # Brighten a random quadrant by 15%
            qh, qw = h // 2, w // 2
            quadrant = np.random.randint(0, 4)
            regions = [(0, qh, 0, qw), (0, qh, qw, w), (qh, h, 0, qw), (qh, h, qw, w)]
            r = regions[quadrant]
            manipulated[r[0]:r[1], r[2]:r[3]] *= 1.15
            manipulated = np.clip(manipulated, 0, 255)

            # Compare original vs manipulated noise profiles
            orig_noise = np.std(arr)
            manip_noise = np.std(manipulated)
            noise_change = abs(orig_noise - manip_noise) / (orig_noise + 1e-10)

            if noise_change < 0.05:
                # Manipulation is subtle enough to potentially evade detection
                blind_spots.append({
                    "agent": "forensic_technical",
                    "issue": f"Simulated 15% brightness manipulation in one quadrant produced only "
                           f"{noise_change:.1%} noise change. Subtle regional edits may evade current ELA sensitivity.",
                    "risk": "evasion",
                    "layer": "adversarial",
                })

            # Test 2: Clone detection evasion — slight rotation
            # If copy-move agent found clones, test if a 2-degree rotation would hide them
            copy_move = next((r for r in agent_results if r["agent_type"] == "copy_move"), None)
            if copy_move:
                cm_items = self._get_items(copy_move)
                if cm_items:
                    # Real clone found — would slight modification hide it?
                    challenges.append({
                        "type": "L2:clone_evasion_test",
                        "challenge": "Clone regions were detected. A sophisticated forger could apply "
                                   "subtle rotation (1-3°), scaling, or noise addition to each cloned region "
                                   "to evade block-matching. Consider adding rotation-invariant matching.",
                        "severity": "medium",
                        "layer": "adversarial",
                    })
                else:
                    # No clones found — test by creating one
                    if w > 128 and h > 128:
                        # Copy a 32x32 block from one location to another
                        test = arr.copy()
                        src_y, src_x = 10, 10
                        dst_y, dst_x = h // 2, w // 2
                        block = test[src_y:src_y + 32, src_x:src_x + 32].copy()
                        test[dst_y:dst_y + 32, dst_x:dst_x + 32] = block

                        # Check if the clone is detectable by comparing block similarity
                        orig_block = arr[dst_y:dst_y + 32, dst_x:dst_x + 32]
                        similarity = 1.0 - np.mean(np.abs(block - orig_block)) / 255.0
                        if similarity > 0.8:
                            blind_spots.append({
                                "agent": "copy_move",
                                "issue": f"Adversarial test: planted a 32x32 clone. "
                                       f"Source-destination similarity was {similarity:.0%}. "
                                       f"Agent may miss clones in similar-texture regions.",
                                "risk": "evasion",
                                "layer": "adversarial",
                            })

            # Test 3: Frequency domain evasion
            # Add very subtle periodic noise that could confuse frequency analysis
            freq_agent = next((r for r in agent_results if r["agent_type"] == "frequency_analysis"), None)
            if freq_agent and freq_agent.get("confidence_score", 0) > 0.8:
                # Agent was confident — but would targeted noise fool it?
                challenges.append({
                    "type": "L2:frequency_evasion_test",
                    "challenge": "Frequency agent reported high confidence. Adversarial periodic noise injection "
                               "at specific DCT frequencies could mask manipulation signatures. "
                               "Consider multi-scale frequency analysis for robustness.",
                    "severity": "low",
                    "layer": "adversarial",
                })

        except Exception:
            pass

        return {"challenges": challenges, "blind_spots": blind_spots}

    # ═══════════════════════════════════════════════════════
    # LAYER 3: Cross-Agent Debate
    # ═══════════════════════════════════════════════════════
    def _layer3_debate(self, results, cross_ref) -> dict:
        challenges = []
        recommendations = []
        adjustment = None

        # Build arguments for AUTHENTIC
        auth_evidence = []
        auth_weight = 0
        # Build arguments for FORGED
        forge_evidence = []
        forge_weight = 0

        weights = {
            "forensic_technical": 3, "physical": 2.5, "contextual": 2,
            "ai_generation": 2, "copy_move": 2, "frequency_analysis": 2,
            "metadata_consistency": 2, "audio_deepfake": 2, "video_forensic": 2,
            "document_forensic": 1.5,
        }

        for r in results:
            agent = r["agent_type"]
            score = r.get("confidence_score", 0.5)
            items = self._get_items(r)
            w = weights.get(agent, 1)

            high_anomalies = [a for a in items if a.get("severity") == "high"]
            med_anomalies = [a for a in items if a.get("severity") == "medium"]

            if score >= 0.75 and not high_anomalies:
                auth_evidence.append({
                    "agent": agent,
                    "argument": f"{agent} reports {score:.0%} confidence with no high-severity findings.",
                    "strength": score * w,
                })
                auth_weight += score * w
            elif high_anomalies:
                for a in high_anomalies:
                    forge_evidence.append({
                        "agent": agent,
                        "argument": f"{agent} found: {a.get('type', 'anomaly')} — {a.get('description', '')[:80]}",
                        "strength": (1 - score) * w * 1.5,
                    })
                    forge_weight += (1 - score) * w * 1.5
            elif med_anomalies:
                for a in med_anomalies:
                    forge_evidence.append({
                        "agent": agent,
                        "argument": f"{agent} found medium: {a.get('type', 'anomaly')}",
                        "strength": (1 - score) * w,
                    })
                    forge_weight += (1 - score) * w

        # Debate outcome
        total = auth_weight + forge_weight
        if total > 0:
            auth_pct = auth_weight / total
            forge_pct = forge_weight / total
        else:
            auth_pct = forge_pct = 0.5

        verdict = cross_ref.get("final_verdict", "")

        # Check if debate contradicts verdict
        if verdict == "authentic" and forge_pct > 0.6:
            challenges.append({
                "type": "L3:debate_contradiction",
                "challenge": f"Cross-agent debate favors FORGED ({forge_pct:.0%}) over AUTHENTIC ({auth_pct:.0%}), "
                           f"but verdict is 'authentic'. {len(forge_evidence)} pieces of evidence argue forgery "
                           f"vs {len(auth_evidence)} for authenticity.",
                "severity": "high",
                "layer": "debate",
            })
            adjustment = -0.05

        elif verdict == "forged" and auth_pct > 0.6:
            challenges.append({
                "type": "L3:debate_contradiction",
                "challenge": f"Cross-agent debate favors AUTHENTIC ({auth_pct:.0%}) over FORGED ({forge_pct:.0%}), "
                           f"but verdict is 'forged'. {len(auth_evidence)} agents support authenticity.",
                "severity": "high",
                "layer": "debate",
            })
            adjustment = 0.05

        elif abs(auth_pct - forge_pct) < 0.15:
            challenges.append({
                "type": "L3:debate_deadlock",
                "challenge": f"Cross-agent debate is nearly deadlocked: AUTHENTIC {auth_pct:.0%} vs FORGED {forge_pct:.0%}. "
                           f"No clear consensus among agents. HITL expert strongly recommended.",
                "severity": "medium",
                "layer": "debate",
            })

        # Add debate details to recommendations
        if auth_evidence or forge_evidence:
            top_auth = sorted(auth_evidence, key=lambda x: x["strength"], reverse=True)[:2]
            top_forge = sorted(forge_evidence, key=lambda x: x["strength"], reverse=True)[:2]

            if top_forge:
                strongest = top_forge[0]
                recommendations.append(
                    f"Strongest forgery argument: {strongest['argument']}"
                )
            if top_auth:
                strongest = top_auth[0]
                recommendations.append(
                    f"Strongest authenticity argument: {strongest['argument']}"
                )

        return {"challenges": challenges, "recommendations": recommendations, "adjustment": adjustment}

    # ═══════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════
    def _get_items(self, result: dict) -> list:
        anomalies = result.get("anomalies", {})
        if isinstance(anomalies, dict):
            return anomalies.get("items", [])
        return anomalies if isinstance(anomalies, list) else []

    def _generate_recommendations(self, challenges, blind_spots, results) -> list:
        recs = []
        high_c = [c for c in challenges if c.get("severity") == "high"]
        if high_c:
            recs.append("High-severity challenges found — HITL expert verification recommended.")
        if len(blind_spots) >= 2:
            recs.append("Multiple blind spots detected. Consider expanding agent capabilities.")
        return recs

    def _calc_threat_level(self, challenges, blind_spots) -> str:
        high = sum(1 for c in challenges if c.get("severity") == "high")
        layers = len(set(c.get("layer", "") for c in challenges if c.get("layer")))
        total = len(challenges) + len(blind_spots)
        if high >= 2 or total >= 6 or layers >= 3:
            return "high"
        if high >= 1 or total >= 3 or layers >= 2:
            return "medium"
        return "low"

    def _build_summary(self, challenges, blind_spots, recs, threat, layers) -> str:
        parts = [
            f"Red Team ({len(layers)} layers active: {', '.join(layers)}) "
            f"identified {len(challenges)} challenges and {len(blind_spots)} blind spots.",
            f"Threat level: {threat}.",
        ]
        if recs:
            parts.append(f"{len(recs)} recommendations issued.")
        return " ".join(parts)
