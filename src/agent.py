
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
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
