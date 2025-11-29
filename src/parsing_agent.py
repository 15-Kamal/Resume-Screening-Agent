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

    try:
        prompt = f"""
You are a highly skilled Job Description Analyzer.

Extract the following information from this job description:

1. Job Title  
2. Must-Have Skills  
3. Good-To-Have Skills  
4. Minimum Years of Experience  
5. Core Responsibilities  

Return ONLY JSON.

Job Description:
{jd_text}
"""

        response = GEMINI_CLIENT.models.generate(
            model="gemini-1.5-flash",
            contents=prompt,
            config=types.GenerateConfig(
                response_mime_type="application/json",
                response_schema=JobRequirements,
            ),
        )

        return JobRequirements.model_validate_json(response.text)

    except Exception as e:
        print(f"[JD Parsing Error] {e}")
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

    try:
        prompt = f"""
You are an expert Resume Analyzer.

Extract the following from this resume:

1. Candidate Name  
2. Total Experience (in years)  
3. Skills  
4. Work Experience Summary  

Return ONLY JSON.

Resume File Name: {file_name}

Resume Text:
{resume_text}
"""

        response = GEMINI_CLIENT.models.generate(
            model="gemini-1.5-flash",
            contents=prompt,
            config=types.GenerateConfig(
                response_mime_type="application/json",
                response_schema=CandidateProfile,
            ),
        )

        return CandidateProfile.model_validate_json(response.text)

    except Exception as e:
        import traceback
        print(f"\n[Resume Parsing Error: {file_name}]")
        print("LLM ERROR:", e)
        print(traceback.format_exc())

        return CandidateProfile(
            candidate_name=file_name,
            total_experience_years=0,
            skills=[],
            work_experience_summary=f"Parsing Error: {e}",
        )
