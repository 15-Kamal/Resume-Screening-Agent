Project Implementation: The Resume Screening Agent

The solution implemented a modern AI Agent based on a modular architecture, with extra emphasis on strong data parsing and an application layer that is resilient. This whole system will be deployed as one independent Python application on Streamlit Cloud.

1. Overview of Technology Stack

  | Category | Technology/Tool | Specific Use in Project |
  | Foundation & Hosting | Python | It is the main programming language used for all the back-end and front-end logic.
  Streamlit: It is the framework applied to the development of a web-based, interactive UI meant for inputting and displaying results.
  | AI Intelligence | Google Gemini API (`google-genai`) | The Large Language Model (LLM) service used for all analysis, extraction, and compare tasks. |
  | | *`gemini-1.5-flash` | Specific model selected for this particular work, due to its strong performance on complex reasoning and structured JSON output.
  Data Structuring Pydantic Define explicit, strict Python types (`JobRequirements`, `CandidateProfile`) that can be used as the schema throughout the Gemini API to ensure consistency in the data.
  Data Handling Pandas Used to organize and display the final results of the comparison in tabular, human-readable format (`st.dataframe`).

Resilience & Parsing | `re` (Regular Expressions) | Used in that very final, critical step of aggressively cleaning raw LLM output to strip non-JSON characters and guarantee successful Pydantic parsing.

2. Implementation Steps and Design Choices

The project was implemented to follow a structured agent pipeline, which ensured the conversion of unstructured text data to measurable results.

A. Data Modeling (Pydantic)
  Action: Two central Pydantic models were defined: `JobRequirements` and `CandidateProfile`.
  Choice Rationale: This was necessary to produce structured output. These were important because calling `.model_json_schema()` on these models instructed the Gemini model to return data in a predictable, verifiable format.

B. The Parsing Agents (`src/parsing_agent.py`)
  Action: Two different agent functions were implemented, which are parse_job_description and parse_candidate_profile.
  Key Technique The method, `GEMINI_CLIENT.models.generate_content`, was called in each function, and the Pydantic JSON schema was passed via the `config.response_schema` parameter.
  Resilience Layer: This module was made robust against errors that popped up during deployment:
  Fixed UnboundLocalError: Inited 'response = None' before trying/excpet blocks to maintain the integrity of the scope of variables.
  JSON Corruption Fix: Added aggressive prompt hardening with STRICT INSTRUCTIONS and a final regex cleaning step, ensuring that only clean JSON is passed to Pydantic.

C. The Evaluation Agent (`app.py` logic)
  Action: The primary logic and assessment are done in the high-level Streamlit application.
  Pipeline:

  1. Input: JD text and uploaded resume files (text).
  2. Parsing: Calls `parse_job_description` once, and `parse_candidate_profile` for each resume. 
  3. Scoring: A separate function calculates a weighted match score based on the extracted `must_have_skills` and `min_years_experience`. 
  4. Rationale Generation: A final, separate LLM call is made to generate a concise narrative summary explaining the fit based on the comparison metrics. 

D. The Frontend (`app.py`) 
  Action: Employed Streamlit components: `st.text_area`, `st.file_uploader`, and `st.dataframe` to handle user interaction. 
  Deployment: The whole application code was deployed on Streamlit Cloud by continuously updating Git to resolve Python runtime problems arising in deployment-specific areas during the whole development.
