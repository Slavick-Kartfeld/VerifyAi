import asyncio
import uuid
import re
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.core.database import get_db
from app.models.models import Case, AgentResult
from app.api.schemas import VerifyResponse, CaseResponse, AgentResultResponse, AnomalyDetail, HITLApprovalRequest
from app.services.storage import compute_sha256, detect_media_type, save_file_locally

router = APIRouter(prefix="/v1", tags=["verify"])

# ── Security constants ────────────────────────────────────────────────────────
MAX_FILE_SIZE   = 50 * 1024 * 1024   # 50 MB
MAX_IMAGE_DIM   = 4096               # px — PIL resize threshold
ANALYSIS_TIMEOUT = 120               # seconds per full analysis
AGENT_TIMEOUT    = 45                # seconds per single agent
CLIENT_ID_RE     = re.compile(r'^[a-zA-Z0-9_\-\.@]{3,128}$')


def _validate_client_id(client_id: str) -> str:
    if not CLIENT_ID_RE.match(client_id):
        raise HTTPException(
            status_code=400,
            detail="client_id must be 3-128 chars: letters, digits, _ - . @"
        )
    return client_id


def _resize_image_if_needed(file_bytes: bytes, filename: str) -> bytes:
    """Resize images larger than MAX_IMAGE_DIM to prevent OOM."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ("jpg", "jpeg", "png", "webp", "bmp", "tiff", "tif"):
        return file_bytes
    try:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(file_bytes))
        w, h = img.size
        if max(w, h) <= MAX_IMAGE_DIM:
            return file_bytes
        # Resize keeping aspect ratio
        ratio  = MAX_IMAGE_DIM / max(w, h)
        new_sz = (int(w * ratio), int(h * ratio))
        img    = img.resize(new_sz, Image.LANCZOS)
        buf    = io.BytesIO()
        fmt    = "JPEG" if ext in ("jpg", "jpeg") else ext.upper()
        if fmt == "JPG":
            fmt = "JPEG"
        img.save(buf, format=fmt)
        return buf.getvalue()
    except Exception:
        return file_bytes   # fallback — return original


@router.post("/verify", response_model=VerifyResponse)
async def submit_verification(
    file: UploadFile = File(...),
    client_id: str = Form(...),
    context: str = Form(default=None),
    db: AsyncSession = Depends(get_db),
):
    """שליחת קובץ מדיה לבדיקת אותנטיות"""

    # ── 1. Validate client_id ──────────────────────────────────────────────
    _validate_client_id(client_id)

    # ── 2. Read & size-check ───────────────────────────────────────────────
    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="הקובץ ריק")
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"קובץ גדול מדי. מקסימום {MAX_FILE_SIZE // (1024*1024)}MB"
        )

    # ── 3. Media type detection ────────────────────────────────────────────
    media_type = detect_media_type(file.filename)
    if media_type == "unknown":
        raise HTTPException(status_code=400, detail="סוג קובץ לא נתמך")

    # ── 4. Resize images to prevent OOM ───────────────────────────────────
    if media_type == "image":
        file_bytes = _resize_image_if_needed(file_bytes, file.filename or "")

    # ── 5. Chain of custody hash ───────────────────────────────────────────
    file_hash = compute_sha256(file_bytes)

    # ── 6. Save file ───────────────────────────────────────────────────────
    file_url = await save_file_locally(file_bytes, file.filename)

    # ── 7. Create case with UUID4 ──────────────────────────────────────────
    case = Case(
        id=str(uuid.uuid4()),
        client_id=client_id,
        media_type=media_type,
        file_url=file_url,
        file_hash=file_hash,
        status="pending",
    )
    db.add(case)
    await db.commit()
    await db.refresh(case)

    # ── 8. Run analysis with hard timeout ─────────────────────────────────
    from app.agents.orchestrator import Orchestrator
    orchestrator = Orchestrator()
    try:
        analysis = await asyncio.wait_for(
            orchestrator.analyze(file_bytes, file.filename, media_type),
            timeout=ANALYSIS_TIMEOUT
        )
    except asyncio.TimeoutError:
        case.status = "timeout"
        case.verdict = "inconclusive"
        case.confidence_score = 0.0
        case.hitl_required = True
        await db.commit()
        raise HTTPException(
            status_code=504,
            detail=f"הניתוח לא הושלם תוך {ANALYSIS_TIMEOUT} שניות. נא לנסות שוב."
        )
    except Exception as e:
        case.status = "error"
        case.verdict = "inconclusive"
        case.confidence_score = 0.0
        case.hitl_required = True
        await db.commit()
        raise HTTPException(status_code=500, detail=f"שגיאה בניתוח: {str(e)[:200]}")

    # ── 9. Save agent results ──────────────────────────────────────────────
    from app.models.models import AgentResult, CrossReferenceResult
    for ar in analysis["agent_results"]:
        agent_result = AgentResult(
            case_id=case.id,
            agent_type=ar["agent_type"],
            findings=ar.get("findings", {}),
            anomalies=ar.get("anomalies", {}),
            confidence_score=ar["confidence_score"],
        )
        db.add(agent_result)

    cross_ref = CrossReferenceResult(
        case_id=case.id,
        combined_score=analysis["confidence_score"],
        reasoning=analysis["cross_reference"]["reasoning"],
        final_verdict=analysis["verdict"],
    )
    db.add(cross_ref)

    # Red Team as AgentResult
    rt = analysis.get("red_team", {})
    if rt:
        rt_result = AgentResult(
            case_id=case.id,
            agent_type="red_team",
            findings={"summary": rt.get("summary", ""), "threat_level": rt.get("threat_level", "low")},
            anomalies={"challenges": rt.get("challenges", []), "blind_spots": rt.get("blind_spots", []), "recommendations": rt.get("recommendations", [])},
            confidence_score=1.0 - abs(rt.get("confidence_adjustment", 0)),
        )
        db.add(rt_result)

    case.status          = "completed"
    case.confidence_score = analysis["confidence_score"]
    case.verdict         = analysis["verdict"]
    case.hitl_required   = analysis["hitl_required"]
    await db.commit()

    return VerifyResponse(
        case_id=case.id,
        status="completed",
        message=f"ניתוח הושלם — {analysis['verdict']}",
    )


@router.get("/verify/{case_id}", response_model=CaseResponse)
async def get_case_status(case_id: str, db: AsyncSession = Depends(get_db)):
    """קבלת סטטוס ותוצאות בדיקה"""
    # Basic UUID format check
    try:
        uuid.UUID(case_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="case_id לא תקין")

    result = await db.execute(select(Case).where(Case.id == case_id))
    case = result.scalar_one_or_none()
    if not case:
        raise HTTPException(status_code=404, detail="מקרה לא נמצא")

    agents_result = await db.execute(
        select(AgentResult).where(AgentResult.case_id == case_id)
    )
    agent_results = agents_result.scalars().all()

    agent_responses = []
    red_team_data = None
    for ar in agent_results:
        if ar.agent_type == "red_team":
            red_team_data = {
                "summary":      ar.findings.get("summary", "")      if ar.findings  else "",
                "threat_level": ar.findings.get("threat_level", "low") if ar.findings else "low",
                "challenges":   ar.anomalies.get("challenges", [])  if ar.anomalies else [],
                "blind_spots":  ar.anomalies.get("blind_spots", []) if ar.anomalies else [],
                "recommendations": ar.anomalies.get("recommendations", []) if ar.anomalies else [],
            }
            continue
        anomalies = [AnomalyDetail(**a) for a in ar.anomalies.get("items", [])] if ar.anomalies else []
        agent_responses.append(AgentResultResponse(
            agent_type=ar.agent_type,
            confidence_score=ar.confidence_score,
            anomalies=anomalies,
            heatmap_url=ar.heatmap_url,
        ))

    hitl_rec = "רמת הביטחון מתחת לסף. מומלץ לשלב מומחה אנושי." if case.hitl_required else None

    resp = CaseResponse(
        id=case.id,
        status=case.status,
        media_type=case.media_type,
        file_hash=case.file_hash,
        confidence_score=case.confidence_score,
        verdict=case.verdict,
        hitl_required=case.hitl_required,
        hitl_recommendation=hitl_rec,
        agent_results=agent_responses,
        created_at=case.created_at,
    )
    result_dict = resp.model_dump()
    if red_team_data:
        result_dict["red_team"] = red_team_data
    return result_dict


@router.post("/verify/{case_id}/hitl")
async def request_hitl(
    case_id: str,
    request: HITLApprovalRequest,
    db: AsyncSession = Depends(get_db),
):
    """אישור הזמנת מומחה אנושי"""
    try:
        uuid.UUID(case_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="case_id לא תקין")

    result = await db.execute(select(Case).where(Case.id == case_id))
    case = result.scalar_one_or_none()
    if not case:
        raise HTTPException(status_code=404, detail="מקרה לא נמצא")
    if not case.hitl_required:
        raise HTTPException(status_code=400, detail="מקרה זה אינו דורש HITL")

    case.status = "hitl_pending"
    await db.commit()
    return {"message": "בקשת HITL התקבלה, מומחה ישובץ בקרוב", "case_id": case_id}


@router.get("/report/{case_id}")
async def download_report(case_id: str, db: AsyncSession = Depends(get_db)):
    """הורדת דוח PDF ראייתי"""
    from fastapi.responses import Response
    from app.models.models import CrossReferenceResult

    try:
        uuid.UUID(case_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="case_id לא תקין")

    result = await db.execute(select(Case).where(Case.id == case_id))
    case = result.scalar_one_or_none()
    if not case:
        raise HTTPException(status_code=404, detail="מקרה לא נמצא")

    agents_result = await db.execute(
        select(AgentResult).where(AgentResult.case_id == case_id)
    )
    agent_results = [
        {"agent_type": ar.agent_type, "confidence_score": ar.confidence_score,
         "findings": ar.findings, "anomalies": ar.anomalies}
        for ar in agents_result.scalars().all()
    ]

    cr_result = await db.execute(
        select(CrossReferenceResult).where(CrossReferenceResult.case_id == case_id)
    )
    cr = cr_result.scalar_one_or_none()

    rt_data = next((ar for ar in agent_results if ar["agent_type"] == "red_team"), None)
    rt_challenges   = rt_data["anomalies"].get("challenges", [])    if rt_data and rt_data.get("anomalies") else []
    rt_blind_spots  = rt_data["anomalies"].get("blind_spots", [])   if rt_data and rt_data.get("anomalies") else []
    rt_recs         = rt_data["anomalies"].get("recommendations", []) if rt_data and rt_data.get("anomalies") else []
    rt_threat       = rt_data["findings"].get("threat_level", "N/A") if rt_data and rt_data.get("findings") else "N/A"

    agent_results = [ar for ar in agent_results if ar["agent_type"] != "red_team"]

    cross_ref = {
        "combined_score": cr.combined_score if cr else 0,
        "reasoning": cr.reasoning if cr else "",
        "final_verdict": cr.final_verdict if cr else "",
        "anomaly_summary": {},
        "red_team_challenges": rt_challenges,
        "red_team_blind_spots": rt_blind_spots,
        "red_team_recommendations": rt_recs,
        "red_team_threat": rt_threat,
    }

    from app.services.report_generator import generate_report
    pdf_bytes = generate_report(
        case_id=case.id,
        verdict=case.verdict or "inconclusive",
        confidence=case.confidence_score or 0,
        file_hash=case.file_hash,
        media_type=case.media_type,
        agent_results=agent_results,
        cross_reference=cross_ref,
    )

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=VerifyAI_Report_{case_id[:8]}.pdf"},
    )
