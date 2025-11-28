import os
from dotenv import load_dotenv
from google.genai import Client
from pydocx import PyDocX
import pypdf

# Load environment variables FIRST
load_dotenv()

# --- 1. Initialize Gemini Client ---
try:
    GEMINI_CLIENT = Client(api_key=os.getenv("GOOGLE_API_KEY"))
except Exception as e:
    print(f"Error initializing Gemini Client: {e}")
    GEMINI_CLIENT = None


# --- 2. Helper for File Reading ---
def extract_text_from_file(file_path: str) -> str:
    """Extracts text content from PDF or DOCX files."""

    file_extension = os.path.splitext(file_path)[1].lower()

    # PDF extract
    if file_extension == ".pdf":
        try:
            reader = pypdf.PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
        except Exception as e:
            return f"Error reading PDF: {e}"

    # DOC or DOCX extract
    elif file_extension in (".docx", ".doc"):
        try:
            html = PyDocX.to_html(file_path)
            cleaned = " ".join(html.split())  # strip HTML formatting
            return cleaned
        except Exception as e:
            return f"Error reading DOCX: {e}"

    else:
        return f"Unsupported file type: {file_extension}"
