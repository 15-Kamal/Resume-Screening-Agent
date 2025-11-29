import streamlit as st
import os
import tempfile
import pandas as pd
from typing import List

# Import all necessary modules from your src directory
# Assuming these imports point to your correctly structured Pydantic models and functions
from src.parsing_agent import parse_job_description, parse_candidate_profile
from src.evaluation_agent import run_evaluation_agent
from src.utils import extract_text_from_file

# --- Configuration ---
st.set_page_config(
    page_title="Gemini Resume Screening Agent",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure uploads folder exists at app start
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR, exist_ok=True)


# --- Helper Function for Running the Core Logic ---
@st.cache_data(show_spinner=False)
def process_resumes(
    jd_text: str, uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]
) -> pd.DataFrame:
    """
    Main orchestration function that runs the parsing and evaluation agents.
    Uses Streamlit's cache to avoid re-running if inputs don't change.

    Important change (Step 3): each uploaded_file is saved to uploads/
    so it can be accessed by file-based parsers and for debugging.
    """
    
    # 1. Parse the Job Description (JD)
    with st.spinner("Analyzing Job Description..."):
        job_requirements = parse_job_description(jd_text)
        
    # Defensive check (your code earlier used "Error" in job_title — preserve behaviour)
    if hasattr(job_requirements, "job_title") and isinstance(job_requirements.job_title, str) and "Error" in job_requirements.job_title:
        st.error(f"Failed to parse Job Description: {job_requirements.core_responsibilities[0]}")
        return pd.DataFrame()

    # --- 🚨 DIAGNOSTIC ADDITION: Display Structured JD ---
    with st.expander("Show Structured Job Requirements (JD)"):
        st.json(job_requirements.model_dump())
    # --------------------------------------------------

    results = []

    # 2. Process each uploaded resume
    for uploaded_file in uploaded_files:
        saved_file_path = None
        try:
            # ---------- Step 3: Save uploaded file to uploads/ ----------
            # Use the original filename but avoid collisions by optionally appending a numeric suffix
            orig_name = uploaded_file.name
            base_name = os.path.splitext(orig_name)[0]
            ext = os.path.splitext(orig_name)[1] or ".pdf"
            save_name = orig_name
            counter = 1
            saved_file_path = os.path.join(UPLOAD_DIR, save_name)
            # If file exists already, append counter
            while os.path.exists(saved_file_path):
                save_name = f"{base_name}_{counter}{ext}"
                saved_file_path = os.path.join(UPLOAD_DIR, save_name)
                counter += 1

            # Write the uploaded bytes to disk (persisted)
            with open(saved_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Optional: Informational log - visible in Streamlit
            st.sidebar.text(f"Saved: {save_name}")

            # ---------- Extract raw text using your utility (file path) ----------
            raw_resume_text = extract_text_from_file(saved_file_path)

            # If the extractor returned empty/very small text, your extract_text_from_file
            # should already contain an OCR fallback. If not, implement OCR in src.utils.
            if not raw_resume_text or len(raw_resume_text.strip()) < 20:
                # small safeguard if extract_text_from_file returns nothing
                # use a simple temp fallback to try reading again (or you can implement OCR in src.utils)
                # For now, we'll raise an explicit error so evaluation picks it up and reports.
                raise ValueError("Extracted resume text is empty (possible scanned PDF). Ensure OCR fallback is present in extract_text_from_file.")

            # Parse Candidate Profile using raw text
            with st.spinner(f"Extracting profile from {save_name}..."):
                candidate_profile = parse_candidate_profile(raw_resume_text, save_name)

            # --- 🚨 DIAGNOSTIC ADDITION: Display Structured Candidate Profile ---
            st.sidebar.markdown(f"### {candidate_profile.candidate_name} Profile Check")
            st.sidebar.json(candidate_profile.model_dump())
            # ------------------------------------------------------------------

            # Run the main Evaluation Agent
            with st.spinner(f"Evaluating {candidate_profile.candidate_name} against JD..."):
                evaluation_result = run_evaluation_agent(job_requirements, candidate_profile)
                
            # Convert Pydantic result to a dictionary for the final table
            result_dict = {
                "Candidate Name": evaluation_result.candidate_name,
                "Final Score": evaluation_result.final_score,
                "Status": evaluation_result.status,
                "Rationale": evaluation_result.recruiter_rationale,
                "Gaps": "\n".join(evaluation_result.quantitative_gaps),
                "Experience (Yrs)": candidate_profile.total_experience_years
            }
            results.append(result_dict)
            
        except Exception as e:
            # This catches file reading errors, parsing errors, or the LLM evaluation error itself
            results.append({
                "Candidate Name": uploaded_file.name,
                "Final Score": 0,
                "Status": f"Failed (System Error)",
                "Rationale": f"Critical error during processing: {e}",
                "Gaps": "System Failure",
                "Experience (Yrs)": 0
            })
        finally:
            # NOTE: We intentionally DO NOT delete the saved_file_path so you can inspect it later.
            # If you prefer to delete temporary saved uploads after processing, uncomment the lines below:
            # if saved_file_path and os.path.exists(saved_file_path):
            #     os.unlink(saved_file_path)
            pass

    # Convert results list to a DataFrame and sort by score
    df = pd.DataFrame(results).sort_values(by="Final Score", ascending=False)
    return df

# --- Streamlit UI Layout ---

st.title("🤖 Gemini Resume Screening Agent")
st.markdown("Use this tool to automatically rank multiple resumes against a single job description using the Gemini API's structured output capability.")

col_jd, col_upload = st.columns([2, 1])

# Input: Job Description Text Area
with col_jd:
    jd_text = st.text_area(
        "Paste Job Description (JD) Text Here:",
        height=300,
        placeholder="Paste the full text of the job description for the 'Senior AI Engineer' role here..."
    )

# Input: File Uploader
with col_upload:
    st.markdown("### 📄 Upload Resumes")
    uploaded_files = st.file_uploader(
        "Upload PDF/DOCX files (Max 10)",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )
    
    st.info("Files are saved to uploads/ and processed one by one.")

# --- Execution Button ---

if st.button("🚀 Run Screening Agent", type="primary", use_container_width=True):
    if not jd_text:
        st.error("Please paste the Job Description text to proceed.")
    elif not uploaded_files:
        st.error("Please upload at least one resume file (.pdf or .docx).")
    else:
        # Run the core logic
        with st.container():
            st.header("Results")
            final_df = process_resumes(jd_text, uploaded_files)
            
            if not final_df.empty:
                # Display the main table
                st.subheader(f"Top Candidates for: {final_df.iloc[0]['Candidate Name']}")
                st.dataframe(
                    final_df[["Candidate Name", "Final Score", "Status", "Experience (Yrs)"]],
                    hide_index=True,
                    use_container_width=True
                )
                
                st.divider()

                # Display detailed results for each candidate
                st.subheader("Detailed Evaluation Rationale")
                for index, row in final_df.iterrows():
                    # Status is checked for the red/green dot color
                    color = "green" if row['Final Score'] > 75 and "Rejected" not in row['Status'] else "red"
                    
                    with st.expander(f"**{row['Candidate Name']}** | Score: **{row['Final Score']}** | Status: **:{color}[{row['Status']}]**"):
                        st.markdown(f"**Recruiter Rationale:**\n\n{row['Rationale']}")
                        st.markdown("---")
                        st.markdown(f"**Quantitative Gaps Identified:**")
                        st.code(row['Gaps'])
                        
                        # Show raw profile data on demand (re-reads saved file from uploads/)
                        if st.checkbox(f"Show Raw Profile Data for {row['Candidate Name']}", key=f"raw_data_{index}"):
                            try:
                                saved_path = os.path.join(UPLOAD_DIR, row['Candidate Name'])
                                # If file was renamed due to collision, try to find the actual file in uploads/
                                if not os.path.exists(saved_path):
                                    # fallback: find first filename that startswith candidate name base
                                    base = os.path.splitext(row['Candidate Name'])[0]
                                    found = None
                                    for fn in os.listdir(UPLOAD_DIR):
                                        if fn.startswith(base):
                                            found = os.path.join(UPLOAD_DIR, fn)
                                            break
                                    if found:
                                        saved_path = found

                                raw_resume_text = extract_text_from_file(saved_path)
                                candidate_profile = parse_candidate_profile(raw_resume_text, row['Candidate Name'])
                                st.json(candidate_profile.model_dump())
                            except Exception:
                                st.warning("Could not display raw profile data.")

