import json
import re
import traceback
from typing import Any, Optional

from google.genai import types
from src.models import JobRequirements, CandidateProfile
from src.utils import GEMINI_CLIENT


# ---------------------------
# Helper utilities
# ---------------------------
def extract_response_text(response: Any) -> Optional[str]:
    """Extract textual content from various Gemini SDK response shapes."""
    if not response:
        return None

    # common simple property
    if hasattr(response, "text") and response.text:
        return response.text

    # shape: response.candidates[0].content or candidate.content
    try:
        if getattr(response, "candidates", None):
            cand0 = response.candidates[0]
            # object-like or dict-like
            return getattr(cand0, "content", None) or cand0.get("content")
    except Exception:
        pass

    # shape: response.output[0].get("content") or object-like
    try:
        if getattr(response, "output", None):
            out0 = response.output[0]
            return getattr(out0, "content", None) or (out0.get("content") if isinstance(out0, dict) else None)
    except Exception:
        pass

    # last-resort attribute checks
    if hasattr(response, "content") and response.content:
        return response.content

    return None


def safe_parse_json(text: str) -> Optional[dict]:
    """Try to parse JSON robustly. Strips surrounding text and tries substrings."""
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try to find the first JSON object in the string
    first_obj = text.find("{")
    last_obj = text.rfind("}")
    if first_obj != -1 and last_obj != -1 and last_obj > first_obj:
        candidate = text[first_obj : last_obj + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # Try to find a top-level array
    first_arr = text.find("[")
    last_arr = text.rfind("]")
    if first_arr != -1 and last_arr != -1 and last_arr > first_arr:
        candidate = text[first_arr : last_arr + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    return None


def heuristic_extract_core_responsibilities_from_text(jd_text: str) -> list:
    """Fallback: pick likely bullet lines from 'Responsibilities' section or first lines."""
    if not jd_text:
        return []

    # attempt to find a responsibilities-like header and extract following bullets
    hdr_patterns = [
        r"(Core\s+Responsibilities|Responsibilities|Key\s+Responsibilities|Duties)\s*[:\-]?\s*",
    ]
    for pat in hdr_patterns:
        m = re.search(pat, jd_text, flags=re.IGNORECASE)
        if m:
            start = m.end()
            snippet = jd_text[start : start + 1000]  # take chunk after header
            lines = [ln.strip() for ln in snippet.splitlines() if ln.strip()]
            bullets = []
            for ln in lines:
                # bullet-like or short sentence
                if re.match(r"^[-•*]\s+", ln):
                    bullets.append(re.sub(r"^[-•*]\s*", "", ln))
                elif len(ln.split()) >= 3:
                    bullets.append(ln)
                if len(bullets) >= 8:
                    break
            if bullets:
                return bullets

    # fallback: return first 3-6 non-empty lines from the JD as responsibilities
    lines = [ln.strip() for ln in jd_text.splitlines() if ln.strip()]
    return lines[:6] if lines else []


def heuristic_summarize_resume(resume_text: str, max_lines: int = 6) -> str:
    """Return a quick summary from the top of the resume text as fallback."""
    if not resume_text:
        return "No resume text to summarize."
    lines = [ln.strip() for ln in resume_text.splitlines() if ln.strip()]
    return " ".join(lines[:max_lines]) if lines else "No resume text to summarize."


# -------------------------------
# Job Description Parsing Agent
# -------------------------------
def parse_job_description(jd_text: str) -> JobRequirements:
    """Uses Gemini to extract structured requirements from JD text."""
    if not GEMINI_CLIENT:
        return JobRequirements(
            job_title="Error",
            must_have_skills=[],
            good_to_have_skills=[],
            min_years_experience=0,
            core_responsibilities=["Gemini client not initialized"],
        )

    response = None
    try:
        prompt = f"""
You are a highly skilled Job Description Analyzer.

Extract the following information from this job description:

1. Job Title
2. Must-Have Skills
3. Good-To-Have Skills
4. Minimum Years of Experience
5. Core Responsibilities

Return ONLY JSON that conforms strictly to the provided schema.

Job Description:
{jd_text}
"""

        response = GEMINI_CLIENT.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt,
            config=types.GenerateConfig(
                response_mime_type="application/json",
                response_schema=JobRequirements.model_json_schema(),
            ),
        )

        # robust extraction
        response_text = extract_response_text(response)
        if not response_text:
            raise ValueError("No textual response returned by Gemini client.")

        # try JSON parse (safe)
        parsed = safe_parse_json(response_text)
        if parsed is not None:
            # prefer Pydantic validation via raw json string, fallback to dict validation
            try:
                return JobRequirements.model_validate_json(response_text)
            except Exception:
                try:
                    return JobRequirements.model_validate(parsed)
                except Exception as e:
                    print("[Job Parsing] validation from dict failed:", e)

        # If we reached here, strict parse/validation failed. Try heuristics:
        print("[Job Parsing] JSON parse/validation failed - attempting heuristic fallback.")
        fallback_core = heuristic_extract_core_responsibilities_from_text(jd_text)
        return JobRequirements(
            job_title="(parsed with fallback)",
            must_have_skills=[],
            good_to_have_skills=[],
            min_years_experience=0,
            core_responsibilities=fallback_core or ["Could not recover responsibilities"],
        )

    except Exception as e:
        # debug printing of response shape & excerpt
        raw_text = extract_response_text(response) or getattr(response, "text", None) or repr(response)
        print("[JD Parsing Error]", e)
        print("Raw LLM Output (excerpt):", str(raw_text)[:800])
        print("Traceback:")
        traceback.print_exc()

        # fallback heuristic
        fallback_core = heuristic_extract_core_responsibilities_from_text(jd_text)
        return JobRequirements(
            job_title="Parsing Failed",
            must_have_skills=[],
            good_to_have_skills=[],
            min_years_experience=0,
            core_responsibilities=fallback_core or [f"Error: {e}"],
        )


# -------------------------------
# Candidate Resume Parsing Agent
# -------------------------------
def parse_candidate_profile(resume_text: str, file_name: str) -> CandidateProfile:
    """Uses Gemini to extract structured profile data from resume text."""
    if not GEMINI_CLIENT:
        return CandidateProfile(
            candidate_name="Error",
            total_experience_years=0,
            skills=[],
            work_experience_summary="Gemini client not initialized",
        )

    response = None
    try:
        prompt = f"""
You are an expert Resume Analyzer.

Extract the following from this resume:

1. Candidate Name
2. Total Experience (in years, e.g., 5.5)
3. Skills
4. Work Experience Summary

Return ONLY JSON that conforms strictly to the provided schema.

Resume File Name: {file_name}

Resume Text:
{resume_text}
"""

        response = GEMINI_CLIENT.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt,
            config=types.GenerateConfig(
                response_mime_type="application/json",
                response_schema=CandidateProfile.model_json_schema(),
            ),
        )

        response_text = extract_response_text(response)
        if not response_text:
            raise ValueError("No textual response returned by Gemini client.")

        parsed = safe_parse_json(response_text)
        if parsed is not None:
            try:
                return CandidateProfile.model_validate_json(response_text)
            except Exception:
                try:
                    return CandidateProfile.model_validate(parsed)
                except Exception as e:
                    print("[Resume Parsing] validation from dict failed:", e)

        # fallback: heuristic summary from resume text
        print("[Resume Parsing] JSON parse/validation failed - using heuristic summary fallback.")
        summary = heuristic_summarize_resume(resume_text)
        return CandidateProfile(
            candidate_name=file_name,
            total_experience_years=0,
            skills=[],
            work_experience_summary=summary,
        )

    except Exception as e:
        raw_text = extract_response_text(response) or getattr(response, "text", None) or repr(response)
        print(f"\n[Resume Parsing Error: {file_name}]")
        print("LLM ERROR:", e)
        print("Raw LLM Output (excerpt):", str(raw_text)[:800])
        traceback.print_exc()

        summary = heuristic_summarize_resume(resume_text)
        return CandidateProfile(
            candidate_name=file_name,
            total_experience_years=0,
            skills=[],
            work_experience_summary=f"Parsing Error: {e}. Fallback summary: {summary}",
        )
