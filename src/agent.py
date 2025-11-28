
import os
from dotenv import load_dotenv
from google.genai import types
from google.genai.client import Client
import os

GEMINI_CLIENT = Client(api_key=os.getenv("GOOGLE_API_KEY"))

from pydocx import PyDocX
import pypdf
import json

from src.models import JobRequirements, CandidateProfile, EvaluationResult

# Load environment variables from .env file
load_dotenv()

# --- 1. Initialize Gemini Client ---
try:
    GEMINI_CLIENT = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
except Exception as e:
    print(f"Error initializing Gemini Client: {e}")
    GEMINI_CLIENT = None

# --- 2. Helper for File Reading ---
def extract_text_from_file(file_path: str) -> str:
    """Extracts text content from PDF or DOCX files."""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.pdf':
        try:
            reader = pypdf.PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
        except Exception as e:
            return f"Error reading PDF: {e}"
            
    elif file_extension in ('.docx', '.doc'):
        try:
            # Note: PyDocX is simple but effective for basic text extraction
            html = PyDocX.to_html(file_path)
            # Simple way to strip HTML tags for plain text
            return ' '.join(html.split()) 
        except Exception as e:
            return f"Error reading DOCX: {e}"
            
    else:
        return f"Unsupported file type: {file_extension}"

# --- 3. Job Description Parsing Agent ---
def parse_job_description(jd_text: str) -> JobRequirements:
    """Uses Gemini to extract structured requirements from raw JD text."""
    if not GEMINI_CLIENT:
        return JobRequirements(job_title="Error", must_have_skills=[], good_to_have_skills=[], min_years_experience=0, core_responsibilities=[])

    try:
        prompt = (
            "You are an expert Job Analyst. Your task is to analyze the following "
            "Job Description text and extract the key requirements into the specified JSON format. "
            "Ensure the output strictly follows the Pydantic schema provided."
            "\n\n--- JOB DESCRIPTION ---\n"
            f"{jd_text}"
        )
        
        # Call Gemini with the Pydantic schema for structured output
        response = GEMINI_CLIENT.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=JobRequirements,
            ),
        )
        
        # The response.text is a JSON string conforming to the schema
        return JobRequirements.model_validate_json(response.text)
        
    except Exception as e:
        print(f"Error in JD parsing: {e}")
        return JobRequirements(job_title="Parsing Failed", must_have_skills=[], good_to_have_skills=[], min_years_experience=0, core_responsibilities=[f"Error: {e}"])


# --- 4. Resume Data Extraction Agent ---
def parse_candidate_profile(resume_text: str, file_name: str) -> CandidateProfile:
    """Uses Gemini to extract structured profile data from raw resume text."""
    if not GEMINI_CLIENT:
        return CandidateProfile(candidate_name="Error", total_experience_years=0, skills=[], work_experience_summary="Error")
        
    try:
        prompt = (
            "You are an expert Resume Data Extractor. Your task is to analyze the following "
            "resume text and structure the key information into the specified JSON format. "
            "Use the provided text to calculate the total years of work experience."
            "\n\n--- RESUME CONTENT ---\n"
            f"{resume_text}"
        )

        response = GEMINI_CLIENT.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=CandidateProfile,
            ),
        )

        # The response.text is a JSON string conforming to the schema
        profile = CandidateProfile.model_validate_json(response.text)
        
        # Attempt to get a name if the LLM missed it, using the filename as a fallback
        if not profile.candidate_name and file_name:
            profile.candidate_name = os.path.splitext(file_name)[0]
            
        return profile
        
    except Exception as e:
        print(f"Error in Candidate Profile parsing for {file_name}: {e}")
        return CandidateProfile(candidate_name=file_name, total_experience_years=0, skills=[], work_experience_summary=f"Parsing Error: {e}")
