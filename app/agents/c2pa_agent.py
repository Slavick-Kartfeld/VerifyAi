"""
C2PA Provenance Agent — reads Content Credentials embedded in media files.

C2PA (Coalition for Content Provenance and Authenticity) is an open standard
by Adobe, Microsoft, Google, Intel, BBC et al. that cryptographically signs
media at the moment of creation and records every subsequent edit.

This agent:
  1. Attempts to read the C2PA manifest from the file bytes.
  2. If found — validates the signature chain and extracts provenance data.
  3. Returns a structured finding: VERIFIED / TAMPERED / NOT_PRESENT.

Supported formats (via c2pa-python SDK): JPEG, PNG, WEBP, TIFF, MP4, MOV,
MP3, WAV, PDF, DOCX and more.
"""

import io
import json
from typing import Any

AGENT_TYPE = "c2pa_provenance"

# MIME type map for c2pa.Reader
_EXT_TO_MIME = {
    "jpg": "image/jpeg", "jpeg": "image/jpeg",
    "png": "image/png",  "webp": "image/webp",
    "tiff": "image/tiff","tif": "image/tiff",
    "gif": "image/gif",
    "mp4": "video/mp4",  "mov": "video/quicktime",
    "avi": "video/avi",  "mkv": "video/x-matroska",
    "mp3": "audio/mpeg", "wav": "audio/wav",
    "flac": "audio/flac","m4a": "audio/mp4",
    "ogg": "audio/ogg",  "opus": "audio/ogg",
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


class C2PAProvenanceAgent:

    AGENT_TYPE = "c2pa_provenance"

    async def analyze(self, file_bytes: bytes, filename: str) -> dict:
        ext  = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        mime = _EXT_TO_MIME.get(ext)

        if not mime:
            return self._result(
                confidence=0.5,
                status="NOT_SUPPORTED",
                findings={"reason": f"C2PA not supported for .{ext} files."},
                anomalies=[],
            )

        try:
            import c2pa
        except ImportError:
            return self._result(
                confidence=0.5,
                status="UNAVAILABLE",
                findings={"reason": "c2pa-python library not installed."},
                anomalies=[],
            )

        # ── Attempt to read manifest ──────────────────────────────────────────
        try:
            stream = io.BytesIO(file_bytes)
            with c2pa.Reader(mime, stream) as reader:
                manifest_json = reader.json()
                manifest = json.loads(manifest_json) if isinstance(manifest_json, str) else manifest_json

            return self._parse_manifest(manifest)

        except Exception as e:
            err = str(e)

            # C2PA not present in this file — normal for most files today
            if any(k in err.lower() for k in ("no active manifest", "not found", "could not parse", "jumbf")):
                return self._result(
                    confidence=0.5,
                    status="NOT_PRESENT",
                    findings={
                        "status":  "No C2PA Content Credentials found in this file.",
                        "meaning": "Most files today lack C2PA. Absence does not indicate forgery — "
                                   "it means provenance cannot be cryptographically verified.",
                    },
                    anomalies=[],
                )

            # Unexpected error
            return self._result(
                confidence=0.5,
                status="ERROR",
                findings={"error": err[:200]},
                anomalies=[],
            )

    # ── Manifest parsing ──────────────────────────────────────────────────────

    def _parse_manifest(self, manifest: dict) -> dict:
        """
        Extract useful provenance data and check for validation errors.
        """
        findings: dict[str, Any] = {"status": "VERIFIED", "raw_manifest": {}}
        anomalies = []

        active_label   = manifest.get("active_manifest", "")
        manifests_dict = manifest.get("manifests", {})
        active         = manifests_dict.get(active_label, {})

        # ── Validation status ─────────────────────────────────────────────────
        validation_status = manifest.get("validation_status", [])
        if isinstance(validation_status, list) and len(validation_status) > 0:
            # Any entry in validation_status = problem
            errors = [v for v in validation_status if isinstance(v, dict)]
            if errors:
                findings["status"] = "TAMPERED"
                for err in errors:
                    anomalies.append({
                        "type":        "C2PA Validation Failure",
                        "description": f"C2PA signature invalid: {err.get('explanation', err.get('code','unknown'))}. "
                                       f"The file may have been modified after signing.",
                        "severity":    "high",
                        "location":    {"x": 50, "y": 10},
                    })

        # ── Provenance data ───────────────────────────────────────────────────
        claim_generator = active.get("claim_generator_info", [{}])
        if isinstance(claim_generator, list) and claim_generator:
            gen = claim_generator[0]
            findings["created_by"] = gen.get("name", "unknown")
            findings["generator_version"] = gen.get("version", "")

        # Signature info
        sig_info = active.get("signature_info", {})
        if sig_info:
            findings["signed_by"]  = sig_info.get("issuer", "unknown")
            findings["signed_at"]  = sig_info.get("time", "unknown")
            findings["cert_serial"] = sig_info.get("cert_serial_number", "")

        # Assertions — actions taken on the file
        assertions = active.get("assertions", [])
        actions = []
        for a in assertions:
            if a.get("label", "").startswith("c2pa.actions"):
                for act in a.get("data", {}).get("actions", []):
                    actions.append({
                        "action":    act.get("action", ""),
                        "when":      act.get("when", ""),
                        "software":  act.get("softwareAgent", ""),
                    })
        if actions:
            findings["edit_history"] = actions
            # Edits are normal — only flag if signature is also broken
            if findings["status"] == "VERIFIED":
                findings["edit_count"] = len(actions)

        # AI generation flag
        ai_assertions = [a for a in assertions if "ai" in a.get("label", "").lower() or
                         "generative" in str(a).lower()]
        if ai_assertions:
            findings["ai_generated"] = True
            anomalies.append({
                "type":        "AI-Generated Content (C2PA)",
                "description": "C2PA manifest declares this file as AI-generated content. "
                               "This is a cryptographically-signed declaration from the creation tool.",
                "severity":    "medium",
                "location":    {"x": 50, "y": 50},
            })

        # ── Confidence ────────────────────────────────────────────────────────
        if findings["status"] == "TAMPERED":
            confidence = 0.15   # very suspicious
        elif findings["status"] == "VERIFIED" and not anomalies:
            confidence = 0.97   # cryptographic proof — highest possible
        else:
            confidence = 0.70

        findings["raw_manifest"] = {
            "active_label": active_label,
            "assertion_count": len(assertions),
        }

        return self._result(confidence, findings["status"], findings, anomalies)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _result(self, confidence: float, status: str, findings: dict, anomalies: list) -> dict:
        return {
            "agent_type":       self.AGENT_TYPE,
            "confidence_score": round(confidence, 2),
            "findings":         {"c2pa_status": status, **findings},
            "anomalies":        {"items": anomalies},
        }
