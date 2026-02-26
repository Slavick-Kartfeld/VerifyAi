from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.core.database import get_db
from app.models.models import Case, AgentResult
from app.api.schemas import VerifyResponse, CaseResponse, AgentResultResponse, AnomalyDetail, HITLApprovalRequest
from app.services.storage import compute_sha256, detect_media_type, save_file_locally

router = APIRouter(prefix="/v1", tags=["verify"])


@router.post("/verify", response_model=VerifyResponse)
async def submit_verification(
    file: UploadFile = File(...),
    client_id: str = Form(...),
    context: str = Form(default=None),
    db: AsyncSession = Depends(get_db),
):
    """שליחת קובץ מדיה לבדיקת אותנטיות"""
    file_bytes = await file.read()

    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="הקובץ ריק")

    # chain of custody — hash מיידי
    file_hash = compute_sha256(file_bytes)
    media_type = detect_media_type(file.filename)

    if media_type == "unknown":
        raise HTTPException(status_code=400, detail="סוג קובץ לא נתמך")

    # שמירת הקובץ
    file_url = await save_file_locally(file_bytes, file.filename)

    # יצירת מקרה בדיקה
    case = Case(
        client_id=client_id,
        media_type=media_type,
        file_url=file_url,
        file_hash=file_hash,
        status="pending",
    )
    db.add(case)
    await db.commit()
    await db.refresh(case)

    # הפעלת Orchestrator
    from app.agents.orchestrator import Orchestrator
    orchestrator = Orchestrator()
    analysis = await orchestrator.analyze(file_bytes, file.filename, media_type)

    # שמירת תוצאות סוכנים
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

    # שמירת הצלבה
    cross_ref = CrossReferenceResult(
        case_id=case.id,
        combined_score=analysis["confidence_score"],
        reasoning=analysis["cross_reference"]["reasoning"],
        final_verdict=analysis["verdict"],
    )
    db.add(cross_ref)

    # שמירת Red Team כ-AgentResult
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

    # עדכון המקרה
    case.status = "completed"
    case.confidence_score = analysis["confidence_score"]
    case.verdict = analysis["verdict"]
    case.hitl_required = analysis["hitl_required"]
    await db.commit()

    return VerifyResponse(
        case_id=case.id,
        status="completed",
        message=f"ניתוח הושלם — {analysis['verdict']}",
    )


@router.get("/verify/{case_id}", response_model=CaseResponse)
async def get_case_status(case_id: str, db: AsyncSession = Depends(get_db)):
    """קבלת סטטוס ותוצאות בדיקה"""
    result = await db.execute(select(Case).where(Case.id == case_id))
    case = result.scalar_one_or_none()

    if not case:
        raise HTTPException(status_code=404, detail="מקרה לא נמצא")

    # טעינת ממצאי סוכנים
    agents_result = await db.execute(
        select(AgentResult).where(AgentResult.case_id == case_id)
    )
    agent_results = agents_result.scalars().all()

    agent_responses = []
    red_team_data = None
    for ar in agent_results:
        if ar.agent_type == "red_team":
            # Red Team מוחזר בנפרד
            red_team_data = {
                "summary": ar.findings.get("summary", "") if ar.findings else "",
                "threat_level": ar.findings.get("threat_level", "low") if ar.findings else "low",
                "challenges": ar.anomalies.get("challenges", []) if ar.anomalies else [],
                "blind_spots": ar.anomalies.get("blind_spots", []) if ar.anomalies else [],
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

    hitl_rec = None
    if case.hitl_required:
        hitl_rec = "רמת הביטחון מתחת לסף. מומלץ לשלב מומחה אנושי."

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

    # הוספת red_team מחוץ ל-schema
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
    result = await db.execute(select(Case).where(Case.id == case_id))
    case = result.scalar_one_or_none()

    if not case:
        raise HTTPException(status_code=404, detail="מקרה לא נמצא")
    if not case.hitl_required:
        raise HTTPException(status_code=400, detail="מקרה זה אינו דורש HITL")

    # TODO: ספרינט 4 — שיבוץ מומחה מתאים
    case.status = "hitl_pending"
    await db.commit()

    return {"message": "בקשת HITL התקבלה, מומחה ישובץ בקרוב", "case_id": case_id}


@router.get("/report/{case_id}")
async def download_report(case_id: str, db: AsyncSession = Depends(get_db)):
    """הורדת דוח PDF ראייתי"""
    from fastapi.responses import Response
    from app.models.models import CrossReferenceResult

    result = await db.execute(select(Case).where(Case.id == case_id))
    case = result.scalar_one_or_none()
    if not case:
        raise HTTPException(status_code=404, detail="מקרה לא נמצא")

    # Agent results
    agents_result = await db.execute(
        select(AgentResult).where(AgentResult.case_id == case_id)
    )
    agent_results = [
        {
            "agent_type": ar.agent_type,
            "confidence_score": ar.confidence_score,
            "findings": ar.findings,
            "anomalies": ar.anomalies,
        }
        for ar in agents_result.scalars().all()
    ]

    # Cross reference
    cr_result = await db.execute(
        select(CrossReferenceResult).where(CrossReferenceResult.case_id == case_id)
    )
    cr = cr_result.scalar_one_or_none()
    cross_ref = {
        "combined_score": cr.combined_score if cr else 0,
        "reasoning": cr.reasoning if cr else "",
        "final_verdict": cr.final_verdict if cr else "",
        "anomaly_summary": {},
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
