from pydantic import BaseModel, Field
from typing import List

# --- 1. Job Description Schema ---
class JobRequirements(BaseModel):
    """Structured requirements extracted from the Job Description."""
    job_title: str = Field(description="The exact title of the role.")
    must_have_skills: List[str] = Field(description="5-10 essential hard skills (e.g., Python, SQL).")
    good_to_have_skills: List[str] = Field(description="3-5 bonus or preferred skills (e.g., Leadership, AWS).")
    min_years_experience: float = Field(description="Minimum required years of professional experience.")
    core_responsibilities: List[str] = Field(description="3-5 main duties/tasks for the role.")

# --- 2. Candidate Profile Schema (Structured Resume Data) ---
class CandidateProfile(BaseModel):
    """Structured data extracted from the Candidate Resume."""
    candidate_name: str
    total_experience_years: float = Field(description="Total years of work experience, calculated from employment dates.")
    skills: List[str] = Field(description="All hard and soft skills found in the resume.")
    work_experience_summary: str = Field(description="A concise summary of the candidate's professional history.")

# --- 3. Final Evaluation Schema ---
class EvaluationResult(BaseModel):
    """The final structured output for the resume evaluation and ranking."""
    candidate_name: str
    final_score: int = Field(description="The matching score from 0 (poor fit) to 100 (perfect fit).")
    status: str = Field(description="Final decision: 'Accepted' (if score > 75) or 'Rejected'.")
    quantitative_gaps: List[str] = Field(description="A list of specific, objective requirements the candidate failed to meet.")
    recruiter_rationale: str = Field(description="A concise, human-readable summary of the final decision and justification.")
