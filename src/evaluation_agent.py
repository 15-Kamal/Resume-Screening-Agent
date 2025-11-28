
import json
from google.genai import types

from src.models import JobRequirements, CandidateProfile, EvaluationResult
from src.utils import GEMINI_CLIENT

def run_evaluation_agent(
    job_reqs: JobRequirements, 
    candidate_profile: CandidateProfile
) -> EvaluationResult:
    """
    The core agent that compares structured job requirements 
    with the candidate profile to generate a score and rationale.
    """
    if not GEMINI_CLIENT:
        return EvaluationResult(
            candidate_name=candidate_profile.candidate_name,
            final_score=0,
            status="Rejected (System Error)",
            quantitative_gaps=["Gemini Client not initialized."],
            recruiter_rationale="System initialization failed."
        )

    # Convert Pydantic objects to JSON strings for embedding into the prompt
    job_reqs_json = json.dumps(job_reqs.model_dump())
    candidate_profile_json = json.dumps(candidate_profile.model_dump())
    
    # --- The Master Prompt: Chain-of-Thought Evaluation ---
    prompt = f"""
    You are a **Senior Technical Recruiter** and an expert in candidate screening. 
    Your task is to objectively evaluate a candidate's profile against the given job requirements.
    
    You must perform a detailed, step-by-step assessment before giving the final result.
    
    **Evaluation Criteria:**
    1. **Quantitative Check (50% Score Weight):** - Must-Have Skills: Did the candidate mention all skills in `must_have_skills`?
       - Experience: Does `total_experience_years` meet or exceed `min_years_experience`?
       - Core Responsibilities: Are the candidate's past work activities (in `work_experience_summary`) highly relevant to the `core_responsibilities`?
       
    2. **Qualitative Check (50% Score Weight):**
       - Good-to-Have Skills: Does the candidate possess skills from `good_to_have_skills`?
       - Transferability & Depth: Assess the overall quality, relevance, and depth of the candidate's experience. Look for indicators of leadership, project ownership, and real-world impact.
       
    **Instructions for Output:**
    - Calculate a `final_score` from 0 to 100.
    - Set `status` to 'Accepted' if the score is above 75, or 'Rejected' otherwise.
    - Fill the `quantitative_gaps` list with specific, failed checks (e.g., "Missing skill: Python 5+ years").
    - Provide a professional, concise `recruiter_rationale` justifying the final score and decision.
    - The output **MUST** strictly adhere to the `EvaluationResult` JSON schema.

    --- JOB REQUIREMENTS ---
    {job_reqs_json}

    --- CANDIDATE PROFILE ---
    {candidate_profile_json}
    """
    
    try:
        response = GEMINI_CLIENT.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=EvaluationResult,
            ),
        )
        
        return EvaluationResult.model_validate_json(response.text)
        
    except Exception as e:
        print(f"Error in Evaluation Agent for {candidate_profile.candidate_name}: {e}")
        return EvaluationResult(
            candidate_name=candidate_profile.candidate_name,
            final_score=0,
            status="Rejected (LLM Error)",
            quantitative_gaps=[f"LLM Processing Error: {e}"],
            recruiter_rationale="The AI failed to generate an evaluation. Please check the API."
        )
