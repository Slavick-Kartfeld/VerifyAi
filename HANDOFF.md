# VerifyAI — Project Handoff Document
# Use this file as context when continuing development in a new Claude session.
# Copy the ENTIRE content of this file as the first message to Claude Sonnet.

---

## SYSTEM PROMPT / CONTEXT

You are continuing development of **VerifyAI**, a multi-agent forensic media authentication platform. I am the founder. We have been building this system together across multiple sessions. You are my CTO, architect, and full-stack developer. You know every file, every agent, every decision.

The system is **live** at: https://verifyai-3.onrender.com
GitHub repo: https://github.com/Slavick-Kartfeld/VerifyAi
Deployment: Render.com (free tier, Python 3.11)
AI API: Anthropic Claude Sonnet 4 (ANTHROPIC_API_KEY set in Render environment)

---

## WHAT WE BUILT (complete history)

### Architecture
- **Backend**: Python 3.11, FastAPI, SQLAlchemy async (aiosqlite for dev, asyncpg for prod)
- **Frontend**: Single-page HTML/CSS/JS with full EN/HE i18n system, SVG icons throughout
- **AI**: Claude Sonnet 4 Vision API for image/audio/video/document analysis
- **Database**: SQLite (dev/Render free), PostgreSQL-ready (config auto-converts URL)
- **Deployment**: Render.com with Procfile, nixpacks.toml, runtime.txt (Python 3.11)

### 10 Specialized Agents
Each agent has an `async analyze(file_bytes, filename) -> dict` method returning `{agent_type, confidence_score, findings, anomalies}`.

1. **Forensic-Technical** (`forensic_agent.py`) — LOCAL, no API
   - ELA analysis (4x4 grid, compression level comparison)
   - EXIF metadata extraction (editing software, conflicting dates)
   - JPEG double-compression detection
   - AI-typical dimension detection (512x512, 1024x1024)

2. **Physical** (`vision_agents.py`) — Claude Vision API + mock fallback
   - Shadow direction, lighting consistency, perspective, reflections, proportions

3. **Contextual** (`vision_agents.py`) — Claude Vision API + mock fallback
   - Historical anachronisms: uniforms, weapons, technology, architecture, vegetation

4. **AI Generation** (`vision_agents.py`) — Claude Vision API + mock fallback
   - Detects DALL-E, Midjourney, Stable Diffusion, Firefly signatures

5. **Copy-Move** (`copy_move_agent.py`) — LOCAL, no API
   - Block matching with 14-dimension DCT features, cosine similarity (threshold 0.92)
   - Minimum shift 30px, merge nearby matches

6. **Frequency Analysis** (`frequency_agent.py`) — LOCAL, no API
   - FFT spectral analysis (radial energy decay)
   - JPEG 8x8 grid alignment check
   - Laplacian high-frequency energy map
   - Noise consistency across quadrants

7. **Metadata Consistency** (`metadata_agent.py`) — LOCAL, no API
   - Date conflicts (DateTime vs DateTimeOriginal)
   - Camera model vs resolution plausibility
   - Software field (Photoshop, AI tools)
   - GPS date vs EXIF date mismatch
   - Thumbnail aspect ratio mismatch
   - ICC profile analysis

8. **Audio Deepfake** (`audio_agent.py`) — LOCAL + Claude API
   - WAV parser, silence ratio, frequency cutoff (>8kHz), spectral CV, zero-crossing rate
   - Claude API for deeper analysis from spectral summary

9. **Video Forensic** (`video_agent.py`) — LOCAL + Claude Vision
   - JPEG frame extraction from video container (needs ffmpeg for production)
   - Temporal brightness consistency, frame similarity, resolution consistency
   - Claude Vision on suspicious frames

10. **Document Forensic** (`document_agent.py`) — LOCAL + Claude API
    - PDF: metadata, creator/producer mismatch, %%EOF count, JavaScript, fonts, signatures
    - DOCX: ZIP structure, core.xml, author mismatch, revision count, macros, external refs
    - TXT: hidden characters, encoding consistency

### Orchestrator (`orchestrator.py`)
4-stage pipeline:
1. Select agents by media type → run in parallel (asyncio.gather)
2. Cross-Reference Engine: weighted scoring
3. Red Team V2: 4-layer adversarial challenge
4. Final verdict with confidence adjustment

Agent selection:
- Image → Forensic, Physical, Contextual, AI-Gen, Copy-Move, Frequency, Metadata (7 agents)
- Video → Forensic, Physical, Contextual, Video, Frequency (5 agents)
- Audio → Forensic, Audio Deepfake (2 agents)
- Document → Forensic, Contextual, Document, Metadata (4 agents)

### Cross-Reference Engine (`cross_reference.py`)
Weights: forensic 18%, physical 13%, contextual 11%, AI-gen 11%, copy-move 9%, frequency 9%, metadata 9%, audio 7%, video 7%, document 6%.
Verdict logic: forged if ≥3 high severity OR (≥2 high + score <0.5). Authentic if score ≥0.75 AND 0 high. Otherwise inconclusive.

### Red Team V2 (`red_team_agent.py`) — 4 layers, LOCAL, no API
- **Layer 0 (Rules)**: False positive/negative checks, consistency gaps, verdict challenges
- **Layer 1 (Statistical Learning)**: Per-agent baselines from history, z-score outlier detection, trend analysis, verdict distribution monitoring. Gets smarter with every analysis.
- **Layer 2 (Adversarial Testing)**: Applies actual manipulations to image (brightness, clone planting), tests if agents would catch them. Identifies evasion blind spots.
- **Layer 3 (Cross-Agent Debate)**: Builds weighted pro-authentic vs pro-forged arguments from all agent findings. Detects when debate contradicts verdict. Reports strongest arguments from each side.

### Frontend (`index.html`)
- Full EN/HE i18n with 120+ translation keys
- Language toggle button in header
- Dark forensic aesthetic with SVG icons (NO emojis anywhere)
- 4-step wizard: Identification → SMS Verify → Upload → Analysis
- Person/Company forms with ID/passport/registration
- SMS verification (currently demo, Twilio-ready)
- Circular SVG progress with dynamic agent grid (shows only relevant agents per media type)
- Interactive anomaly heatmap with markers and tooltips
- Agent result cards with SVG icons and confidence bars
- Red Team panel with challenges, blind spots, recommendations
- Cryptographic stamp (SHA-256 via WebCrypto)
- PDF download button
- System summary with 4 metric cards

### Landing Page (`landing.html`)
- Route: `/` (marketing), app at `/app`
- Hero section, Problem ($8.2B), Technology (9 agents), Differentiators, CTA
- Scroll animations via IntersectionObserver

### API Routes (`routes.py`)
- `POST /v1/verify` — Upload file, run full pipeline, return case_id
- `GET /v1/verify/{case_id}` — Get results + red team data
- `POST /v1/verify/{case_id}/hitl` — Request HITL expert
- `GET /v1/report/{case_id}` — Download PDF report
- `GET /` — Landing page
- `GET /app` — Analysis platform
- `GET /health` — Health check
- `GET /docs` — Swagger (auto-generated by FastAPI)

### PDF Report (`report_generator.py`)
ReportLab-generated with: header, requester info, verdict (colored box), file info, agent details tables, agent confidence overview, Red Team report section, cross-reference, cryptographic stamp (SHA-256), legal disclaimer.

### Database Models (`models.py`)
- `Case`: id (UUID), client_id, media_type, file_url, file_hash, status, verdict, confidence_score, hitl_required, created_at
- `AgentResult`: id, case_id FK, agent_type, findings JSON, anomalies JSON, confidence_score, heatmap_url
- `CrossReferenceResult`: id, case_id FK, combined_score, reasoning, final_verdict

### Vision API Integration (`vision_agents.py`)
Priority chain: Claude → OpenAI → Mock
- `_call_claude_vision()`: Anthropic API with base64 image, auto media type detection
- `_call_openai_vision()`: OpenAI fallback
- Prompts in English, responses in Hebrew or English depending on context

---

## KNOWN VULNERABILITIES (from our assessment)

**CRITICAL (Sprint 1 priority):**
- No authentication (anyone can use the system and drain API credits)
- SQLite in production (data lost every deploy)
- No file validation (no size limit, no magic byte check)
- Ephemeral file storage (uploads vanish on redeploy)

**HIGH:**
- SMS is demo (accepts any 4-digit code)
- No HTTPS enforcement
- Mock data looks like real data (no DEMO MODE indicator)
- Video agent needs ffmpeg (currently extracts JPEG markers only)
- Audio agent only supports WAV format

**MEDIUM:**
- No logging/monitoring (no Sentry, no structlog)
- Red Team learning is in-memory (resets on restart)
- No agent performance metrics
- No batch processing
- No GDPR compliance

---

## ACTION PLAN (4 sprints, 8 weeks)

**Sprint 1 (Week 1-2) — SECURE:**
JWT auth, rate limiting, PostgreSQL, R2 file storage, file validation, HTTPS, Twilio SMS

**Sprint 2 (Week 3-4) — REAL:**
ffmpeg, audio conversion, DEMO MODE banner, logging, Sentry, Red Team to DB

**Sprint 3 (Week 5-6) — RELIABLE:**
Agent metrics, batch upload, webhooks, versioning, spending caps

**Sprint 4 (Week 7-8) — SELLABLE:**
Admin dashboard, API docs, GDPR delete, multi-tenant, SSO

---

## DOCUMENTS CREATED
1. System Documentation (complete build history)
2. Reconstruction Prompt (for rebuilding from scratch)
3. Vulnerability Assessment (security + technical debt)
4. MVP to Production Roadmap (agents, data, security, revenue)
5. Cost Breakdown (3 scenarios: $5, $78/mo, $1,385/mo)
6. Action Plan (4 sprints, 28 tasks, risk matrix)
7. Investor One-Pager
8. NDA for investors

---

## UX NOTES TO REMEMBER
- Icons need to remain SVG throughout (we spent effort removing all emojis)
- Landing page needs improvement (noted for later)
- All text must be bilingual (EN/HE) via the i18n system
- Agent animation grid must show only relevant agents per media type

---

## HOW TO WORK WITH ME
- I prefer Hebrew communication but technical content in English
- Show me options before building (use selection widgets)
- Plan before coding — discuss architecture first
- Deploy to Render via: git add . && git commit -m "msg" && git push
- ZIP files for me when needed — I extract and overwrite (be careful with .git folder)
- Keep costs minimal — I'm bootstrapping

---

## CURRENT FILE STRUCTURE

```
verifyai/
├── app/
│   ├── agents/
│   │   ├── audio_agent.py          # Audio deepfake (local + Claude)
│   │   ├── copy_move_agent.py      # Clone detection (local)
│   │   ├── cross_reference.py      # Weighted scoring engine
│   │   ├── document_agent.py       # PDF/DOCX/TXT analysis
│   │   ├── forensic_agent.py       # ELA/EXIF/compression (local)
│   │   ├── frequency_agent.py      # Spectral/DCT/noise (local)
│   │   ├── metadata_agent.py       # EXIF consistency (local)
│   │   ├── orchestrator.py         # 4-stage pipeline
│   │   ├── red_team_agent.py       # 4-layer adversarial (local)
│   │   ├── video_agent.py          # Frame analysis (local + Claude)
│   │   └── vision_agents.py        # Physical/Contextual/AI-Gen (Claude)
│   ├── api/
│   │   ├── routes.py               # FastAPI endpoints
│   │   └── schemas.py              # Pydantic models
│   ├── core/
│   │   ├── config.py               # Settings + DB URL conversion
│   │   ├── database.py             # SQLAlchemy async engine
│   │   └── celery_app.py           # Celery (placeholder)
│   ├── models/
│   │   └── models.py               # Case, AgentResult, CrossReferenceResult
│   ├── services/
│   │   ├── report_generator.py     # PDF generation (ReportLab)
│   │   └── storage.py              # SHA-256, media detection, file save
│   ├── static/
│   │   ├── index.html              # Main app (EN/HE, SVG icons)
│   │   └── landing.html            # Marketing page
│   └── main.py                     # FastAPI app + routes
├── requirements.txt
├── Procfile
├── runtime.txt                     # Python 3.11.7
├── nixpacks.toml
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .gitignore
└── DEPLOY.md
```

---

## RESUME WORKING

I'm ready to continue building. The ZIP file attached contains the complete current codebase. Pick up exactly where we left off. Our next priorities from the action plan are Sprint 1 security tasks, but I may have other requests first. Ask me what I want to work on.
