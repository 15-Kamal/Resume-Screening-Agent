import json  # <--- ADDED for schema handling (optional)
from google.genai import types
from src.models import JobRequirements, CandidateProfile
from src.utils import GEMINI_CLIENT  # Correct client import


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

    response = None  # <-- ensure defined even if exception occurs
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

        response = GEMINI_CLIENT.models.generate_content(  # <--- generate_content
            model="gemini-1.5-flash",
            contents=prompt,
            config=types.GenerateConfig(
                response_mime_type="application/json",
                response_schema=JobRequirements.model_json_schema(),
            ),
        )

        # robust extraction of text from SDK response
        response_text = None
        if hasattr(response, "text") and response.text:
            response_text = response.text
        elif getattr(response, "candidates", None):
            # some SDK shapes put content in candidates[0].content
            response_text = getattr(response.candidates[0], "content", None)
        elif getattr(response, "output", None):
            # other SDK shapes use output list/dicts
            try:
                response_text = response.output[0].get("content")
            except Exception:
                response_text = None

        if not response_text:
            raise ValueError("No textual response returned by Gemini client.")

        return JobRequirements.model_validate_json(response_text)

    except Exception as e:
        raw_text = getattr(response, "text", None)
        # try some fallbacks for raw_text for debugging
        if not raw_text and getattr(response, "candidates", None):
            raw_text = getattr(response.candidates[0], "content", None)
        if not raw_text and getattr(response, "output", None):
            try:
                raw_text = response.output[0].get("content")
            except Exception:
                raw_text = "No response text available."

        print(f"[JD Parsing Error] {e}")
        print(f"Raw LLM Output: {str(raw_text)[:400]}...")
        return JobRequirements(
            job_title="Parsing Failed",
            must_have_skills=[],
            good_to_have_skills=[],
            min_years_experience=0,
            core_responsibilities=[f"Error: {e}"],
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

    response = None  # <-- ensure defined even if exception occurs
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

        # robust extraction of text from SDK response
        response_text = None
        if hasattr(response, "text") and response.text:
            response_text = response.text
        elif getattr(response, "candidates", None):
            response_text = getattr(response.candidates[0], "content", None)
        elif getattr(response, "output", None):
            try:
                response_text = response.output[0].get("content")
            except Exception:
                response_text = None

        if not response_text:
            raise ValueError("No textual response returned by Gemini client.")

        return CandidateProfile.model_validate_json(response_text)

    except Exception as e:
        import traceback

        raw_text = getattr(response, "text", None)
        if not raw_text and getattr(response, "candidates", None):
            raw_text = getattr(response.candidates[0], "content", None)
        if not raw_text and getattr(response, "output", None):
            try:
                raw_text = response.output[0].get("content")
            except Exception:
                raw_text = "No response text available."

        print(f"\n[Resume Parsing Error: {file_name}]")
        print("LLM ERROR:", e)
        print(f"Raw LLM Output: {str(raw_text)[:400]}...")
        print(traceback.format_exc())

        return CandidateProfile(
            candidate_name=file_name,
            total_experience_years=0,
            skills=[],
            work_experience_summary=f"Parsing Error: {e}",
        )
