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

# --- Helper Function for Running the Core Logic ---

@st.cache_data(show_spinner=False)
def process_resumes(
    jd_text: str, uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]
) -> pd.DataFrame:
    """
    Main orchestration function that runs the parsing and evaluation agents.
    Uses Streamlit's cache to avoid re-running if inputs don't change.
    """
    
    # 1. Parse the Job Description (JD)
    with st.spinner("Analyzing Job Description..."):
        job_requirements = parse_job_description(jd_text)
        
    if "Error" in job_requirements.job_title:
        st.error(f"Failed to parse Job Description: {job_requirements.core_responsibilities[0]}")
        return pd.DataFrame()

    # --- 🚨 DIAGNOSTIC ADDITION: Display Structured JD ---
    with st.expander("Show Structured Job Requirements (JD)"):
        # Display the parsed JD data for verification
        st.json(job_requirements.model_dump())
    # --------------------------------------------------

    results = []
    
    # 2. Process each uploaded resume
    for uploaded_file in uploaded_files:
        temp_file_path = ""
        try:
            # Streamlit UploadedFile object must be written to a temporary file 
            # for your external file readers (pypdf, pydocx) to access the path.
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                tmp.write(uploaded_file.getvalue())
                temp_file_path = tmp.name

            # Extract raw text
            raw_resume_text = extract_text_from_file(temp_file_path)
            
            # Parse Candidate Profile using raw text
            with st.spinner(f"Extracting profile from {uploaded_file.name}..."):
                candidate_profile = parse_candidate_profile(raw_resume_text, uploaded_file.name)
            
            # --- 🚨 DIAGNOSTIC ADDITION: Display Structured Candidate Profile ---
            # Using the sidebar to keep the main view clean
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
            # The next step should be to modify evaluation_agent.py to give a better error message here
            results.append({
                "Candidate Name": uploaded_file.name,
                "Final Score": 0,
                "Status": f"Failed (System Error)",
                "Rationale": f"Critical error during processing: {e}",
                "Gaps": "System Failure",
                "Experience (Yrs)": 0
            })
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

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
    
    st.info("Files are processed one by one using a temporary directory.")

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
                        
                        # The original code to re-parse the profile to show the structured model output 
                        # This section is generally okay, but now we have the data in the sidebar too.
                        if st.checkbox(f"Show Raw Profile Data for {row['Candidate Name']}", key=f"raw_data_{index}"):
                            try:
                                # This is necessary because of Streamlit's file handling: re-access the file content
                                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_files[index].name)[1]) as tmp:
                                    tmp.write(uploaded_files[index].getvalue())
                                    temp_file_path = tmp.name
                                raw_resume_text = extract_text_from_file(temp_file_path)
                                candidate_profile = parse_candidate_profile(raw_resume_text, row['Candidate Name'])
                                st.json(candidate_profile.model_dump())
                            except Exception:
                                st.warning("Could not display raw profile data.")
                            finally:
                                if os.path.exists(temp_file_path):
                                    os.unlink(temp_file_path)
