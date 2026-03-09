"""
Document Forensic Agent — ניתוח מסמכים פורנזי.
מנתח PDF/DOCX/TXT:
- מטאדטה (יוצר, תוכנה, תאריכים)
- עקביות פונטים (ב-PDF)
- חתימות דיגיטליות
- שינויים חשודים
- Claude API לניתוח תוכן (עם API key)
"""
import io
import os
import json
import struct
from datetime import datetime
import httpx


class DocumentForensicAgent:
    """ניתוח פורנזי של מסמכים."""

    AGENT_TYPE = "document_forensic"

    async def analyze(self, file_bytes: bytes, filename: str) -> dict:
        anomalies = []
        findings = {}
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

        findings["extension"] = ext
        findings["file_size"] = len(file_bytes)

        if ext == "pdf":
            result = self._analyze_pdf(file_bytes)
        elif ext in ("doc", "docx"):
            result = self._analyze_docx(file_bytes)
        elif ext == "txt":
            result = self._analyze_txt(file_bytes)
        else:
            result = {"anomalies": [], "findings": {"format": "generic"}}

        anomalies.extend(result.get("anomalies", []))
        findings.update(result.get("findings", {}))

        # Claude API for content analysis
        api_result = await self._claude_content_analysis(file_bytes, ext, findings)
        if api_result:
            anomalies.extend(api_result.get("anomalies", []))
            findings["ai_analysis"] = api_result.get("summary", "")
            findings["source"] = "claude_api"
        else:
            findings["source"] = "local_analysis"

        # Confidence
        high_count = sum(1 for a in anomalies if a["severity"] == "high")
        confidence = max(0.2, 0.88 - high_count * 0.15 - len(anomalies) * 0.05)

        return {
            "agent_type": self.AGENT_TYPE,
            "confidence_score": round(confidence, 2),
            "findings": findings,
            "anomalies": {"items": anomalies},
        }

    def _analyze_pdf(self, file_bytes: bytes) -> dict:
        """Analyze PDF structure and metadata."""
        anomalies = []
        findings = {}

        text = file_bytes.decode("latin-1", errors="ignore")

        # 1. Extract metadata from PDF info dict
        # Look for /Creator, /Producer, /ModDate, /CreationDate
        metadata = {}
        for key in ["Creator", "Producer", "ModDate", "CreationDate", "Author", "Title"]:
            marker = f"/{key}"
            idx = text.find(marker)
            if idx != -1:
                # Extract value — simplified parsing
                start = idx + len(marker)
                # Skip whitespace and opening paren/bracket
                while start < len(text) and text[start] in " \t\n\r(":
                    start += 1
                end = start
                while end < len(text) and end - start < 200 and text[end] not in ")\n>":
                    end += 1
                val = text[start:end].strip()
                if val:
                    metadata[key] = val

        findings["metadata"] = metadata

        # Check for suspicious metadata
        creator = metadata.get("Creator", "")
        producer = metadata.get("Producer", "")

        if creator and producer and creator != producer:
            # Different creator and producer can indicate editing
            findings["creator_producer_mismatch"] = True
            anomalies.append({
                "type": "Metadata Mismatch",
                "description": f"PDF Creator ('{creator[:50]}') differs from Producer ('{producer[:50]}'). "
                             f"This may indicate the document was modified by different software than originally created.",
                "severity": "medium",
                "location": {"x": 50, "y": 10},
            })

        # Check dates
        create_date = metadata.get("CreationDate", "")
        mod_date = metadata.get("ModDate", "")
        if create_date and mod_date and create_date != mod_date:
            findings["dates_differ"] = True
            anomalies.append({
                "type": "Date Discrepancy",
                "description": f"Creation date differs from modification date. "
                             f"Document was modified after initial creation.",
                "severity": "low",
                "location": {"x": 50, "y": 15},
            })

        # 2. Check for incremental updates (multiple %%EOF markers = multiple saves)
        eof_count = text.count("%%EOF")
        findings["eof_count"] = eof_count
        if eof_count > 1:
            anomalies.append({
                "type": "Incremental Updates",
                "description": f"PDF contains {eof_count} revision layers (%%EOF markers). "
                             f"Multiple revisions may hide previous content or indicate tampering.",
                "severity": "high" if eof_count > 3 else "medium",
                "location": {"x": 50, "y": 50},
            })

        # 3. Check for JavaScript (potential malware/manipulation)
        if "/JavaScript" in text or "/JS " in text:
            findings["has_javascript"] = True
            anomalies.append({
                "type": "Embedded JavaScript",
                "description": "PDF contains embedded JavaScript code. "
                             "This is unusual in legitimate documents and may indicate manipulation or malware.",
                "severity": "high",
                "location": {"x": 50, "y": 70},
            })

        # 4. Count fonts — too many different fonts can indicate splicing
        font_count = text.count("/Type /Font")
        findings["font_count"] = font_count
        if font_count > 20:
            anomalies.append({
                "type": "Excessive Fonts",
                "description": f"Document uses {font_count} different fonts. "
                             f"Unusually high font count may indicate content was pasted from multiple sources.",
                "severity": "medium",
                "location": {"x": 50, "y": 30},
            })

        # 5. Check for digital signatures
        has_sig = "/Type /Sig" in text or "/SigFlags" in text
        findings["has_signature"] = has_sig
        if has_sig:
            findings["signature_present"] = True
            # Note: full signature validation requires dedicated crypto libraries

        # 6. Check for form fields (fillable forms can be manipulated)
        form_fields = text.count("/Type /Annot") + text.count("/AcroForm")
        findings["form_fields"] = form_fields

        return {"anomalies": anomalies, "findings": findings}

    def _analyze_docx(self, file_bytes: bytes) -> dict:
        """Analyze DOCX (ZIP-based) structure."""
        anomalies = []
        findings = {}
        import zipfile

        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
                names = z.namelist()
                findings["file_count"] = len(names)

                # Check for core.xml (metadata)
                if "docProps/core.xml" in names:
                    core = z.read("docProps/core.xml").decode("utf-8", errors="ignore")
                    findings["has_core_metadata"] = True

                    # Extract key fields
                    for tag in ["dc:creator", "cp:lastModifiedBy", "dcterms:created", "dcterms:modified", "cp:revision"]:
                        start_tag = f"<{tag}"
                        idx = core.find(start_tag)
                        if idx != -1:
                            val_start = core.find(">", idx) + 1
                            val_end = core.find("<", val_start)
                            if val_start > 0 and val_end > val_start:
                                findings[tag.split(":")[-1]] = core[val_start:val_end].strip()

                    # Check if different people created vs modified
                    creator = findings.get("creator", "")
                    modifier = findings.get("lastModifiedBy", "")
                    if creator and modifier and creator != modifier:
                        anomalies.append({
                            "type": "Author Mismatch",
                            "description": f"Created by '{creator}' but last modified by '{modifier}'. "
                                         f"Different authors may indicate unauthorized editing.",
                            "severity": "medium",
                            "location": {"x": 50, "y": 10},
                        })

                    # High revision count
                    rev = findings.get("revision", "0")
                    try:
                        rev_num = int(rev)
                        if rev_num > 50:
                            anomalies.append({
                                "type": "High Revision Count",
                                "description": f"Document has {rev_num} revisions. Unusually high revision count "
                                             f"may indicate extensive editing or content manipulation.",
                                "severity": "low",
                                "location": {"x": 50, "y": 20},
                            })
                    except ValueError:
                        pass

                # Check for embedded macros
                macro_files = [n for n in names if n.startswith("word/vbaProject") or n.endswith(".bin")]
                if macro_files:
                    findings["has_macros"] = True
                    anomalies.append({
                        "type": "Embedded Macros",
                        "description": "Document contains VBA macros. Macros can be used for malicious purposes "
                                     "and may indicate the document has been tampered with.",
                        "severity": "high",
                        "location": {"x": 50, "y": 60},
                    })

                # Check for external relationships (links to external content)
                if "word/_rels/document.xml.rels" in names:
                    rels = z.read("word/_rels/document.xml.rels").decode("utf-8", errors="ignore")
                    ext_links = rels.count("TargetMode=\"External\"")
                    if ext_links > 0:
                        findings["external_links"] = ext_links
                        anomalies.append({
                            "type": "External References",
                            "description": f"Document contains {ext_links} external references. "
                                         f"External links can be used for tracking or loading remote content.",
                            "severity": "medium" if ext_links > 3 else "low",
                            "location": {"x": 50, "y": 40},
                        })

        except Exception as e:
            findings["parse_error"] = str(e)

        return {"anomalies": anomalies, "findings": findings}

    def _analyze_txt(self, file_bytes: bytes) -> dict:
        """Basic text file analysis."""
        anomalies = []
        findings = {}

        try:
            text = file_bytes.decode("utf-8", errors="replace")
            findings["length"] = len(text)
            findings["lines"] = text.count("\n") + 1

            # Check for hidden characters
            hidden = sum(1 for c in text if ord(c) < 32 and c not in "\n\r\t")
            if hidden > 10:
                findings["hidden_chars"] = hidden
                anomalies.append({
                    "type": "Hidden Characters",
                    "description": f"Text contains {hidden} non-printable control characters. "
                                 f"These may be used to hide information or indicate tampering.",
                    "severity": "medium",
                    "location": {"x": 50, "y": 50},
                })

            # Check encoding consistency
            try:
                file_bytes.decode("utf-8")
                findings["encoding"] = "utf-8"
            except UnicodeDecodeError:
                findings["encoding"] = "mixed"
                anomalies.append({
                    "type": "Encoding Inconsistency",
                    "description": "File contains mixed character encodings. "
                                 "Legitimate documents typically use consistent encoding.",
                    "severity": "low",
                    "location": {"x": 50, "y": 50},
                })

        except Exception as e:
            findings["error"] = str(e)

        return {"anomalies": anomalies, "findings": findings}

    async def _claude_content_analysis(self, file_bytes: bytes, ext: str, findings: dict) -> dict | None:
        """Use Claude to analyze document content for inconsistencies."""
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            return None

        # Build description
        desc = f"Document analysis (.{ext}, {len(file_bytes)} bytes):\n"
        for k, v in findings.items():
            if k not in ("metadata",) and not isinstance(v, (dict, list)):
                desc += f"- {k}: {v}\n"

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": "claude-sonnet-4-20250514",
                        "max_tokens": 1000,
                        "system": "You are a document forensics expert. Based on the metadata and structural analysis provided, assess if this document shows signs of tampering, forgery, or suspicious manipulation. RESPOND ONLY WITH JSON: {\"anomalies\": [{\"type\": \"...\", \"description\": \"...\", \"severity\": \"high/medium/low\"}], \"summary\": \"brief assessment\"}",
                        "messages": [{"role": "user", "content": desc}],
                    }
                )
                data = resp.json()
                if "content" in data:
                    text = data["content"][0].get("text", "")
                    if text.startswith("```"):
                        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
                    parsed = json.loads(text.strip())
                    for a in parsed.get("anomalies", []):
                        a["location"] = a.get("location", {"x": 50, "y": 50})
                    return parsed
        except Exception:
            pass
        return None
