import os
import json
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser


# 1. Configuration

load_dotenv()

st.title("AI Screening Text")

#2.  intialize LLM and Embeddings

llm = ChatOpenAI(
    model = "gpt-5.6-luna")

embeddings = OpenAIEmbeddings(
    model = "text-embedding-3-small"
)

# 3 . Streamlit memory

if 'candidate_memory' not in st.session_state:
    st.session_state.candidate_memory =  {}


if "vectorstores" not in st.session_state:
    st.session_state.vectorstores = {}



# 4. Helper Function

def clean_json_response(result: any) -> dict:
    """
    Converts the parser result into a Python dictionary.
    """

    if isinstance(result, dict):
        return result

    if hasattr(result, "content"):
        result = result.content

    if isinstance(result, str):

        result = result.strip()

        # Remove markdown code fences if the model returned them
        if result.startswith("```json"):
            result = result[7:]

        elif result.startswith("```"):
            result = result[3:]

        if result.endswith("```"):
            result = result[:-3]

        result = result.strip()

        return json.loads(result)

    raise ValueError("Unable to convert LLM response to JSON.")


# 5. PDF Processing

def process_pdf(uploaded_file):
    """
    Loads a PDF and creates a FAISS vector database
    specifically for that candidate.
    """
    # create temporary PDF
    with tempfile.NamedTemporaryFile(
        delete = False,
        suffix = ".pdf",
        ) as temp_file:

        temp_file.write(uploaded_file.getbuffer())
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)

        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200
        )
        chunks = text_splitter.split_documents(documents)

        # Add candidate metadata
        for chunk in chunks:
            chunk.metadata['candidate'] = uploaded_file.name


        vectorstore = FAISS.from_documents(
            chunks,
            embeddings
        )

        return vectorstore, len(documents), len(chunks)

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

#6. retrieval

def retrieve_resume_information(
    vectorstore,
    job_description,
    k=8
):
    """
    retrieves the most relevant resume chunks
    according to the Job Description.
    """

    retirever = vectorstore.as_retriever(
        search_kwargs ={
            "k": k
        }
    )

    documents = retirever.invoke(job_description)

    return documents

#7. format retrived documents
def format_documents(documents):

    formatted = []

    for i, doc in enumerate(documents):

        page_number = doc.metadata.get(
            "page",
            "Unknown"
        )

        formatted.append(
            f"""--- Resume Evidence {i + 1} --- 
            Page: {page_number}
        {doc.page_content}
        """
        )

    return "\n".join(formatted)

#8. Resume Evaluation Prompt

evaluation_prompt = ChatPromptTemplate.from_template(
"""
You are an AI resume analysis assistant.

Your task is to evaluate ONE candidate resume against
the provided Job Description.

IMPORTANT RULES:

1. Use ONLY the resume evidence provided below.
2. Do NOT invent information.
3. Do NOT assume a candidate has a skill that is not documented.
4. If something is not mentioned in the resume, explicitly say:
   "Not mentioned in resume."
5. The Job Description is used to determine what requirements
   should be checked.
6. Do not use outside information about the candidate.
7. Do not make a hiring decision.
8. Provide evidence from the resume whenever possible.

JOB DESCRIPTION:

{job_description}


RESUME EVIDENCE:

{resume_context}


Return ONLY valid JSON.

Required JSON structure:

{{
    "candidate_summary": "Short summary based only on resume evidence",

    "matching_skills": [
        "skill 1",
        "skill 2"
    ],

    "missing_skills": [
        "skill 1",
        "skill 2"
    ],

    "additional_relevant_skills": [
        "skill 1",
        "skill 2"
    ],

    "experience_alignment": [
        "Evidence about relevant experience"
    ],

    "education_alignment": [
        "Evidence about education"
    ],

    "project_alignment": [
        "Evidence about relevant projects"
    ],

    "strengths": [
        "strength 1",
        "strength 2"
    ],

    "weaknesses": [
        "weakness 1",
        "weakness 2"
    ],

    "evidence": [
        "Specific evidence from the resume"
    ],

    "skill_match_percentage": 0,

    "experience_match_percentage": 0,

    "education_match_percentage": 0,

    "project_match_percentage": 0
}}

The percentage fields must be numbers between 0 and 100.
"""
)


#9. Evaluate Candidate

def evaluate_candidate(
        candidate_name,
        job_description,
        vectorstore
):
    
    #Retrieve
    documents = retrieve_resume_information(
        vectorstore,
        job_description,
        k=8
    )

    if not documents:
        raise ValueError(
            "No relevant information was retrieved from the resume."
        )

    resume_context = format_documents(documents)

    # LLM

    chain = evaluation_prompt | llm | JsonOutputParser()

    result = chain.invoke({
        "job_description": job_description,
        "resume_context": resume_context
    })

    result = clean_json_response(result)

    # calculate match score

    skill_score = float(
        result.get(
            "skill_match_percentage",
            0
        )
        )

    experience_score = float(
        result.get(
            "experience_match_percentage",
            0
        ))

    education_score = float(
        result.get(
            "education_match_percentage",
            0
        ))
    
    project_score = float(
        result.get(
            "project_match_percentage",
            0
        )
    )

    # weight score
    # 
    # Skills = 40%
    # Experience = 30%
    # Education = 10%
    # Project = 20%

    match_score = (
        skill_score * 0.40
        + experience_score * 0.30
        + education_score * 0.10
        + project_score * 0.20
    )

    match_score = round(
        min(max(match_score, 0), 100),
        2
    )

    result["candidate"] = candidate_name
    result["match_score"] = match_score

    #Store retrieve evidence as well
    result["retrieved_evidence"] = [
        {
            "page": doc.metadata.get(
                "page",
                "Unkown"
            ),
            "text": doc.page_content
        }
        for doc in documents
    ]
    return result

# 10. multi-candidate Comparision

comparison_prompt = ChatPromptTemplate.from_template(
"""
You are an AI candidate comparison assistant.

Compare multiple candidates against the same Job Description.

IMPORTANT:

1. Use ONLY the candidate evaluation information provided.
2. Do not invent candidate information.
3. Do not introduce outside information.
4. Do not make assumptions.
5. Explain differences using documented evidence.
6. Do not make the final hiring decision.
7. Identify which candidate has the highest calculated
   match score if applicable.
8. The highest score does NOT automatically mean the candidate
   should be hired.

JOB DESCRIPTION:

{job_description}


CANDIDATE EVALUATIONS:

{candidate_data}


Return ONLY valid JSON.

Required structure:

{{
    "comparison_summary": "Overall factual comparison",

    "candidate_comparison": [
        {{
            "candidate": "Resume A.pdf",

            "match_score": 0,

            "key_matching_skills": [],

            "key_missing_skills": [],

            "major_strengths": [],

            "major_weaknesses": [],

            "important_evidence": []
        }}
    ],

    "highest_match_score_candidate":
        "Candidate name",

    "highest_score": 0,

    "important_differences": [],

    "recruiter_considerations": []
}}
"""
)

def compare_candidates(
        job_description,
        candidate_memory
):
    # Remove retrieved chunks from comaprison
    # to keep the comparison concise.

    comparison_data = {}

    for candidate, evaluation in candidate_memory.items():

        comparison_data[candidate] = {
            key: value
            for key, value in evaluation.items()
            if key != "retrieved_evidence"
        }

    candidate_data = json.dumps(
        comparison_data,
        indent = 4
    )
    chain = (
        comparison_prompt
        | llm
        | JsonOutputParser()
    )

    result = chain.invoke(
        {
            "job_description": job_description,
            "candidate_data": candidate_data
        }
    )

    return clean_json_response(result)

# 11. Streamlit UI

st.title('AI-Powered Resume Screening Assistant')

st.write(
    """
Analyze multiple resumes against a Job Description using 
LangChain, RAG, FAISS, and GPT models.
"""
)

st.info(
    """
The system uses the uploaded resumes as its evidence source. 
Information not documented in a resume is treated as 
"Not mentioned in resume."
"""
)

#12. Job Description

st.header("1. Job Description")

job_description = st.text_area(
    "Enter the Job Description",
    height=250,
    placeholder="""
Example:

We are looking for a Data Scientist with experience in:

- Python
- SQL
- Machine Learning
- Pandas
- Scikit-learn
- XGBoost
- Statistics
- AWS
- Data visualization

Bachelor's degree in Computer Science, Data Science,
Statistics or related field.
"""
)

#13. upload resumes

st.header("2. Upload Resumes")

uploaded_files = st.file_uploader(
    "Upload one or more PDF resumes",
    type = ['pdf'],
    accept_multiple_files = True
)

#14. evaluate Button

if st.button(
    "Evaluate Resumes",
    type="primary"
):

    if not job_description.strip():

        st.warning(
            "Please enter a Job Description first."
        )

    elif not uploaded_files:

        st.warning(
            "Please upload at least one resume."
        )

    else:

        progress = st.progress(0)

        total_files = len(uploaded_files)

        for index, uploaded_file in enumerate(
            uploaded_files
        ):

            candidate_name = uploaded_file.name

            try:

                with st.spinner(
                    f"Processing {candidate_name}..."
                ):
                    vectorstore, page_count, chunk_count = (
                        process_pdf(uploaded_file)
                    )

                    # Save vectorstore
                    st.session_state.vectorstores[
                        candidate_name
                    ] = vectorstore

                    
                    evaluation = evaluate_candidate(
                        candidate_name,
                        job_description,
                        vectorstore
                    )

                    st.session_state.candidate_memory[
                        candidate_name
                    ] = evaluation

                st.success(
                    f"{candidate_name} evaluated successfully."
                )

            except Exception as e:

                st.error(
                    f"Error processing {candidate_name}: {str(e)}"
                )

            progress.progress(
                (index + 1) / total_files
            )

        st.success(
            "All available resumes have been processed."
        )


#15. Display Candidate memory
if st.session_state.candidate_memory:
    
    st.divider()

    st.header("3. Candidate Evaluation")

    st.write(
        f"""
        Candidates currently stored in memory:
        **{len(st.session_state.candidate_memory)}**
    """
    )

    # Candidate Selector
    candidate_names = list(
        st.session_state.candidate_memory.keys()
    )

    selected_candidate = st.selectbox(
        "Select a candidate",
        candidate_names
    )

    candidate = st.session_state.candidate_memory[
        selected_candidate
    ]

    st.subheader(
        f"Evaluation: {selected_candidate}"
    )

    score = candidate.get( "match_score", 0)

    st.metric(
        "Match Score",
        f"{score}/100"
    )

    # Summary

    st.subheader("Candidate Summary")

    st.write(
        candidate.get(
            "candidate_summary",
            "Not available."
        )
    )

    #Skills
    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Matching Skills")

        matching_skills = candidate.get(
            "matching_skills",
            []
        )

        if matching_skills:

            for skill in matching_skills:

                st.success(
                    f"✓ {skill}"
                )

        else:

            st.write("None documented.")

    with col2:

        st.subheader("Missing Skills")

        missing_skills = candidate.get(
            "missing_skills",
            []
        )

        if missing_skills:

            for skill in missing_skills:

                st.warning(
                    f"• {skill}"
                )

        else:

            st.write("None identified.")

    #Additional Skills

    st.subheader("Additional Relevant Skills")

    additional_skills = candidate.get(
        "additional_relevant_skills",
        []
    )

    if additional_skills:

        st.write(
            ", ".join(additional_skills)
        )

    else:

        st.write(
            "None documented."
        )

    #Experience

    st.subheader("Experience Alignment")

    for item in candidate.get("experience_alignment", []):
        st.write(
            f". {item}"
        )

    #Projects

    st.subheader("Project ALignment")

    for item in candidate.get("project_alignment", []):
        st.write(f". {item}")

    #Strengths

    st.subheader("Strengths")

    for item in candidate.get("strengths", []):
        st.success(
            f". {item}"
        )

    #Weaknesses

    st.subheader("Weaknesses")

    for item in candidate.get("weaknesses", []):
        st.warning(f"{item}")

    #Evidence
    
    with st.expander("View Resume Evidence"):
        evidence = candidate.get("evidence", [])

        for item in evidence:
            st.write(
                f"-{item}"
            )

    #Scoring breakdown
    with st.expander(
        "View Score Breakdown"
    ):

        st.write(
            f"""
            Skill alignment: 
            {candidate.get("skill_match_percentage", 0)}%

            Experience alignment:
            {candidate.get("experience_match_percentage", 0)}%

            Education alignment:
            {candidate.get("education_match_percentage", 0)}%

            Project alignment:
            {candidate.get("project_match_percentage", 0)}%

            Final weighted score:
            {candidate.get("match_score", 0)}/100
            """
        )
#16. Multi-candidate comparison

if len(st.session_state.candidate_memory) >= 2:
    st.divider()

    st.header(
        "4. Multi-Candidate Comparison"
    ) 

    st.write(
        """
        Compare candidates using the evaluatios stored
        in candidate memory.
    """
    )

    available_candidates = list(
        st.session_state.candidate_memory.keys()
    )

    selected_candidates = st.multiselect(
        'Select candidates to compare',
        available_candidates,
        default = available_candidates
    )
    if st.button(
        "Compare Selected Candidates"
    ):
        if len(selected_candidate) < 2:
            st.warning("Select at Least two candidates.")

        else:
            selected_memory = {
                candidate:
                st.session_state.candidate_memory[
                    candidate
                ]
                for candidate in selected_candidates
            }
            with st.spinner(
                "Comparing candidates..."
            ):
                comparison = compare_candidates(
                    job_description,
                    selected_memory
                )


            st.subheader("Comparison Summary")

            st.write(comparison.get("comparison_summary", " "))

            #Table
            st.subheader(
                "Candidate Comparison"
            )

            comparison_rows = []

            for item in comparison.get(
                "candidate_comparison",
                []
            ):

                comparison_rows.append(
                    {
                        "Candidate":
                            item.get(
                                "candidate",
                                ""
                            ),

                        "Match Score":
                            item.get(
                                "match_score",
                                0
                            ),

                        "Matching Skills":
                            ", ".join(
                                item.get(
                                    "key_matching_skills",
                                    []
                                )
                            ),

                        "Missing Skills":
                            ", ".join(
                                item.get(
                                    "key_missing_skills",
                                    []
                                )
                            )
                    }
                )

            if comparison_rows:

                st.dataframe(
                    comparison_rows,
                    use_container_width=True
                )

            #Highest score
            st.subheader(
                "Highest Calculated Match Score"
            )

            highest_candidate = comparison.get(
                "highest_match_score_candidate",
                "Not available"
            )

            highest_score = comparison.get(
                "highest_score",
                0
            )

            st.info(
                f"""
                Candidate with the highest calculated
                alignment score: **{highest_candidate}**

                Score: **{highest_score}/100**

                This is a comparison of documented alignment
                with the Job Description, not an automated
                hiring decision.
                """
            )

            #Important Difference
            st.subheader(
                "Important Differences"
            )

            for item in comparison.get(
                "important_differences",
                []
            ):

                st.write(
                    f"• {item}"
                )

            #Recruiter Considerations
            st.subheader(
                "Recruiter Considerations"
            )

            for item in comparison.get(
                "recruiter_considerations",
                []
            ):

                st.write(
                    f"• {item}"
                )

#17. Candidate Memory

if st.session_state.candidate_memory:
    st.divider()

    st.header(
        "5. Candidate Memory"
    )

    st.write(
        """
        The application stores structured evaluations
        for the current Streamlit session.
        """
    )

    memory_view = {}

    for candidate_name, evaluation in (
        st.session_state.candidate_memory.items()
    ):

        memory_view[candidate_name] = {
            key: value
            for key, value in evaluation.items()
            if key != "retrieved_evidence"
        }

    st.json(memory_view)

    #Download Memory

    memory_json = json.dumps( memory_view, indent=4)

    st.download_button(
        label="Download Candidate Memory",
        data=memory_json,
        file_name="candidate_memory.json",
        mime="application/json"
    )

# 18. CLEAR MEMORY


st.divider()

if st.button(
    "Clear Candidate Memory"
):

    st.session_state.candidate_memory = {}
    st.session_state.vectorstores = {}

    st.rerun()
