from google.genai import types

from src.models import JobRequirements, CandidateProfile
from src.utils import GEMINI_CLIENT # Import the client

# --- Job Description Parsing Agent ---
def parse_job_description(jd_text: str) -> JobRequirements:
    """Uses Gemini to extract structured requirements from raw JD text."""
    if not GEMINI_CLIENT:
        # ... (error handling code remains the same)
        return JobRequirements(job_title="Error", must_have_skills=[], good_to_have_skills=[], min_years_experience=0, core_responsibilities=[])
    
    try:
        # ... (Gemini call logic remains the same)
        prompt = (
            "You are an expert Job Analyst. Your task is to analyze the following "
            # ... (rest of the prompt)
        )
        
        response = GEMINI_CLIENT.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=JobRequirements,
            ),
        )
        
        return JobRequirements.model_validate_json(response.text)
        
    except Exception as e:
        # ... (error handling code remains the same)
        print(f"Error in JD parsing: {e}")
        return JobRequirements(job_title="Parsing Failed", must_have_skills=[], good_to_have_skills=[], min_years_experience=0, core_responsibilities=[f"Error: {e}"])


# --- Candidate Data Extraction Agent ---
def parse_candidate_profile(resume_text: str, file_name: str) -> CandidateProfile:
    """Uses Gemini to extract structured profile data from raw resume text."""
    if not GEMINI_CLIENT:
        # ... (error handling code remains the same)
        return CandidateProfile(candidate_name="Error", total_experience_years=0, skills=[], work_experience_summary="Error")
        
    try:
        # ... (Gemini call logic remains the same)
        prompt = (
            "You are an expert Resume Data Extractor. Your task is to analyze the following "
            # ... (rest of the prompt)
        )

        response = GEMINI_CLIENT.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=CandidateProfile,
            ),
        )

        # ... (post-processing logic remains the same)
        profile = CandidateProfile.model_validate_json(response.text)
        
        return profile
        
    except Exception as e:
        # ... (error handling code remains the same)
        print(f"Error in Candidate Profile parsing for {file_name}: {e}")
        return CandidateProfile(candidate_name=file_name, total_experience_years=0, skills=[], work_experience_summary=f"Parsing Error: {e}")
