# AI Resume Screening Assistant

An AI-powered Resume Screening Assistant built with **Python, LangChain, FAISS, OpenAI, and Streamlit**.

The application allows users to upload multiple resumes in PDF format, provide a Job Description (JD), and evaluate each candidate based on the information available in their resume.

The project uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant information from resumes before generating a structured candidate evaluation.

---

## Features

* Upload multiple PDF resumes
* Enter a Job Description
* Extract text from PDF resumes
* Split resume text into smaller chunks
* Generate embeddings for resume content
* Store embeddings using FAISS
* Retrieve relevant resume sections using the Job Description
* Analyze candidates using an OpenAI chat model
* Generate structured candidate evaluations
* Calculate a job-description alignment score
* Identify matching skills
* Identify missing skills
* Identify additional relevant skills
* Analyze experience alignment
* Analyze education alignment
* Analyze project alignment
* Generate candidate strengths
* Generate candidate weaknesses
* Provide supporting evidence from the resume
* Store evaluated candidates in session memory
* Compare multiple evaluated candidates
* Display candidate comparison using Streamlit
* Download candidate evaluation data as JSON

---

## Project Architecture

```text
                    Job Description
                           |
                           v
                  +------------------+
                  |   Resume Upload  |
                  +------------------+
                           |
                           v
                    PDF Text Loader
                           |
                           v
                  Text Chunking
                           |
                           v
                  OpenAI Embeddings
                           |
                           v
                     FAISS Vector DB
                           |
                           v
                    Similarity Search
                           |
                           v
                 Relevant Resume Chunks
                           |
                           v
                    OpenAI Chat Model
                           |
                           v
              Structured Candidate Analysis
                           |
                           v
                 Candidate Memory
                           |
                           v
                Multi-Candidate Comparison
                           |
                           v
                    Streamlit UI
```

---

## Technologies Used

| Technology                     | Purpose                         |
| ------------------------------ | ------------------------------- |
| Python                         | Application development         |
| Streamlit                      | Web application interface       |
| LangChain                      | LLM and RAG workflow            |
| OpenAI                         | Embeddings and language model   |
| FAISS                          | Vector similarity search        |
| PyPDF                          | PDF text extraction             |
| RecursiveCharacterTextSplitter | Text chunking                   |
| JSON                           | Structured candidate data       |
| python-dotenv                  | Environment variable management |

---

## Project Structure

```text
AI-Resume-Screening-Assistant/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .env
```

> **Note:** The `.env` file should only exist locally and must not be uploaded to GitHub.

---

# Requirements

Make sure Python is installed on your system.

The project dependencies are provided in:

```text
requirements.txt
```

Install the required packages using:

```bash
pip install -r requirements.txt
```

The project uses the following dependencies:

```text
ipykernel
python-dotenv
langchain_community
langchain_text_splitters
langchain_openai
pypdf
faiss-cpu
streamlit
langchain_core
```

---

# OpenAI API Key Configuration

The application requires an OpenAI API key.

Create a file named:

```text
.env
```

in the root directory of the project.

Add:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Replace `your_openai_api_key_here` with your actual API key.

For security reasons, never upload your API key to GitHub.

Add `.env` to your `.gitignore` file:

```text
.env
```

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/J0YA1/AI-Resume-Screening.git
```

Replace:

```text
J0YA1
```

and

```text
AI_Resume_Screening
```

with your GitHub username and repository name.

---

## 2. Navigate to the Project Directory

```bash
cd AI_Resume_Screening
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure Environment Variables

Create a `.env` file:

```text
.env
```

Add your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

---

## 5. Run the Application

Start the Streamlit application using:

```bash
python -m streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

---

# How the Application Works

## 1. Upload Resumes

The user can upload multiple resumes in PDF format.

For example:

```text
Resume_A.pdf
Resume_B.pdf
Resume_C.pdf
```

Each resume is processed independently.

Each candidate gets their own retrieval process and vector store.

---

## 2. Enter the Job Description

The user provides the Job Description against which the resumes will be evaluated.

Example:

```text
We are looking for a Data Scientist with experience in Python,
Machine Learning, SQL, Pandas, Scikit-learn, and XGBoost.

The candidate should have experience building machine learning
models and working with data preprocessing and model evaluation.
```

---

## 3. Extract Resume Text

The application uses `PyPDFLoader` to extract text from the uploaded PDF.

The general workflow is:

```text
PDF
 |
 v
PyPDFLoader
 |
 v
Extracted Resume Text
```

---

## 4. Split the Resume Text

The extracted resume text is divided into smaller chunks using:

```text
RecursiveCharacterTextSplitter
```

This allows the application to retrieve specific and relevant sections of the resume.

Example:

```text
Resume
   |
   +-- Chunk 1
   +-- Chunk 2
   +-- Chunk 3
   +-- Chunk 4
   +-- ...
```

---

## 5. Generate Embeddings

Each resume chunk is converted into an embedding.

Embeddings represent text as numerical vectors, allowing semantically similar pieces of text to be found.

The application uses an OpenAI embedding model through LangChain.

```text
Resume Chunk
     |
     v
Embedding Model
     |
     v
Vector Representation
```

---

## 6. Store Embeddings in FAISS

The generated embeddings are stored in a FAISS vector database.

Each candidate has a separate vector store.

```text
Candidate A
    |
    +---- FAISS Vector Store A


Candidate B
    |
    +---- FAISS Vector Store B


Candidate C
    |
    +---- FAISS Vector Store C
```

Keeping the vector stores separate helps prevent information from different resumes from being mixed together during retrieval.

---

# Retrieval-Augmented Generation

This project uses a **Retrieval-Augmented Generation (RAG)** pipeline.

The basic workflow is:

```text
Job Description
       |
       v
Retriever
       |
       v
FAISS Vector Store
       |
       v
Relevant Resume Chunks
       |
       v
Prompt
       |
       v
OpenAI Chat Model
       |
       v
Candidate Evaluation
```

Instead of sending the entire resume directly to the model every time, the application retrieves the most relevant resume sections based on the Job Description.

---

# Candidate Evaluation

After retrieving the relevant resume information, the application sends the following information to the language model:

```text
Job Description
+
Retrieved Resume Information
```

The model then generates a structured evaluation.

The evaluation contains information such as:

```json
{
    "candidate_summary": "...",
    "matching_skills": [],
    "missing_skills": [],
    "additional_relevant_skills": [],
    "experience_alignment": "...",
    "education_alignment": "...",
    "project_alignment": "...",
    "strengths": [],
    "weaknesses": [],
    "evidence": []
}
```

The evaluation is intended to be grounded in the uploaded resume and retrieved resume content.

---

# Candidate Match Score

The application calculates an overall alignment score using several dimensions.

The current weighting is:

```text
Skills       → 40%
Experience   → 30%
Education    → 10%
Projects     → 20%
```

For example:

```text
Skills Match       = 90%
Experience Match   = 80%
Education Match    = 100%
Project Match      = 85%
```

The overall score can be calculated as:

```text
Overall Score =
(Skills × 0.40)
+
(Experience × 0.30)
+
(Education × 0.10)
+
(Projects × 0.20)
```

Example:

```text
(90 × 0.40)
+ (80 × 0.30)
+ (100 × 0.10)
+ (85 × 0.20)

= 88%
```

The score represents **alignment with the provided Job Description**. It is not intended to make an automatic hiring decision.

---

# Candidate Memory

The application stores structured candidate evaluations in Streamlit session memory.

For example:

```python
candidate_memory = {
    "Resume_A.pdf": {
        "match_score": 88,
        "matching_skills": [
            "Python",
            "Machine Learning",
            "SQL"
        ],
        "missing_skills": [
            "AWS"
        ],
        "strengths": [
            "Strong machine learning project experience"
        ],
        "weaknesses": [
            "Limited professional experience"
        ]
    }
}
```

This allows the application to evaluate several candidates first and compare their stored evaluation results afterward.

---

# Multi-Candidate Comparison

The application can evaluate multiple resumes and maintain their structured results in memory.

For example:

```text
Resume A
    |
    +-- Evaluation
    |
    +-- Match Score
    |
    +-- Skills
    |
    +-- Experience
    |
    +-- Projects


Resume B
    |
    +-- Evaluation
    |
    +-- Match Score
    |
    +-- Skills
    |
    +-- Experience
    |
    +-- Projects


Resume C
    |
    +-- Evaluation
    |
    +-- Match Score
    |
    +-- Skills
    |
    +-- Experience
    |
    +-- Projects
```

The stored evaluations can then be compared based on:

* Match score
* Skill alignment
* Experience alignment
* Education alignment
* Project relevance
* Matching skills
* Missing skills
* Strengths
* Weaknesses
* Supporting evidence

The comparison provides structured information that can be reviewed by the recruiter or hiring team.

---

# Example Candidate Comparison

Suppose a Job Description requires:

```text
Data Scientist

Required Skills:
- Python
- SQL
- Machine Learning
- Pandas
- Scikit-learn
- XGBoost

Preferred Skills:
- AWS
- Deep Learning
- NLP
```

The application may produce a comparison such as:

| Candidate   | Match Score | Matching Skills                  | Missing Skills |
| ----------- | ----------: | -------------------------------- | -------------- |
| Candidate A |         88% | Python, SQL, ML, Pandas, XGBoost | AWS            |
| Candidate B |         82% | Python, SQL, ML, Scikit-learn    | XGBoost, AWS   |
| Candidate C |         76% | Python, Pandas, ML               | SQL, AWS       |

The table is intended to help users review the documented differences between candidates.

---

# RAG Pipeline in Detail

```text
                    PDF Resume
                        |
                        v
                 +--------------+
                 | PyPDFLoader  |
                 +--------------+
                        |
                        v
                 Extracted Text
                        |
                        v
        +-----------------------------+
        | RecursiveCharacterText      |
        | Splitter                    |
        +-----------------------------+
                        |
                        v
                   Text Chunks
                        |
                        v
              OpenAI Embeddings
                        |
                        v
                +---------------+
                |     FAISS     |
                | Vector Store  |
                +---------------+
                        |
                        v
                    Retriever
                        |
                        |
          +-------------+-------------+
          |                           |
          v                           v
 Job Description              Resume Context
          |                           |
          +-------------+-------------+
                        |
                        v
                  Prompt Template
                        |
                        v
                 OpenAI Chat Model
                        |
                        v
              Structured Evaluation
                        |
                        v
                Candidate Memory
                        |
                        v
              Candidate Comparison
```

---

# Why Separate FAISS Stores?

Each resume is processed independently.

For example:

```text
Resume A → FAISS A
Resume B → FAISS B
Resume C → FAISS C
```

This prevents the retrieval process for Candidate A from accidentally retrieving text belonging to Candidate B.

The architecture therefore maintains candidate-level isolation during the retrieval stage.

---

# Streamlit Interface

The Streamlit application provides an interface for:

```text
+--------------------------------------------------+
|          AI Resume Screening Assistant           |
+--------------------------------------------------+
|                                                  |
| Job Description                                  |
|                                                  |
| [ Enter Job Description ]                        |
|                                                  |
| Upload Resumes                                   |
|                                                  |
| [ Resume A.pdf ]                                 |
| [ Resume B.pdf ]                                 |
| [ Resume C.pdf ]                                 |
|                                                  |
| [ Evaluate Resumes ]                             |
|                                                  |
+--------------------------------------------------+
```

After processing, the application displays candidate evaluations and comparison information.

---

# Example Output

A candidate evaluation can contain:

```text
Candidate Summary
-----------------
Candidate has experience in Python, machine learning,
and data preprocessing with relevant academic projects.

Matching Skills
---------------
- Python
- Machine Learning
- Pandas
- Scikit-learn

Missing Skills
--------------
- AWS

Experience Alignment
--------------------
Relevant experience is present through internships and
machine learning projects.

Education Alignment
-------------------
Educational background is aligned with the technical
requirements of the role.

Project Alignment
-----------------
Projects demonstrate practical experience with machine
learning and model development.

Strengths
---------
- Strong Python knowledge
- Machine learning project experience
- Experience with Scikit-learn

Weaknesses
----------
- Limited professional experience
- No demonstrated AWS experience
```

---

# Session Memory

Candidate evaluations are stored using Streamlit's session state.

Conceptually:

```text
Streamlit Session
       |
       v
candidate_memory
       |
       +---- Candidate A
       |
       +---- Candidate B
       |
       +---- Candidate C
```

This allows candidates evaluated during the same application session to be compared.

### Important

The current implementation uses session memory rather than a permanent database.

Therefore, candidate memory can be lost when the Streamlit session or application is restarted.

The application can provide the structured evaluation data as JSON so that it can be saved externally.

---

# Security

Never commit API keys to GitHub.

Your `.gitignore` should contain:

```text
.env
__pycache__/
*.pyc
.venv/
venv/
```

A typical `.gitignore` file can be:

```gitignore
.env
__pycache__/
*.pyc
.venv/
venv/
```

If an API key is accidentally pushed to a public repository, revoke the exposed key and create a new one.

---

# Limitations

## 1. PDF Text Extraction

The quality of the evaluation depends partly on the quality of the extracted PDF text.

Scanned or image-based PDFs may require OCR, which is not currently included in this version.

---

## 2. LLM Output

Large language models can sometimes produce incorrect or incomplete information.

The generated evaluation should therefore be reviewed by a human.

---

## 3. Resume Information

The system should only use information available in the uploaded resume and retrieved context.

It should not be assumed that a candidate possesses a skill simply because it is common for the role.

---

## 4. Session-Based Memory

Candidate memory currently exists within the Streamlit session.

It is not a permanent database.

---

## 5. API Usage

The application uses API-based language model and embedding services.

API usage may incur costs depending on the configured provider, models, and usage.

---

# Future Improvements

The following features could be added in future versions:

* Persistent candidate database
* PostgreSQL integration
* OCR for scanned resumes
* DOCX resume support
* Automatic Job Description parsing
* Automatic skill extraction
* Skill normalization
* Experience timeline extraction
* Candidate filtering
* Advanced analytics dashboard
* Recruiter authentication
* PDF report generation
* Candidate feedback system
* Resume section classification
* Improved semantic retrieval
* Hybrid keyword + semantic search
* Deployment to Streamlit Cloud
* More advanced candidate comparison

---

# Learning Concepts Demonstrated

This project demonstrates practical implementation of several AI and machine learning concepts:

### Natural Language Processing

* Text extraction
* Text chunking
* Semantic similarity
* Embeddings
* LLM-based analysis

### Retrieval-Augmented Generation

* Document loading
* Text splitting
* Embeddings
* Vector databases
* Similarity search
* Retrieval
* Context injection
* LLM generation

### LangChain

* Document loaders
* Text splitters
* Embeddings
* Vector stores
* Retrievers
* Prompt templates
* Chat models
* Output parsers

### Generative AI

* Prompt engineering
* Structured LLM output
* Candidate evaluation
* Multi-document comparison

### Streamlit

* File upload
* User input
* Session state
* Interactive UI
* Data presentation
* JSON download

---

# Running the Project

The complete workflow is:

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git

# Enter project directory
cd YOUR_REPOSITORY

# Install dependencies
pip install -r requirements.txt

# Create .env
# Add OPENAI_API_KEY

# Run application
python -m streamlit run app.py
```

---

# Git Commands

After creating or modifying the project, you can push it to GitHub using:

```bash
git init
```

```bash
git add .
```

```bash
git commit -m "Initial commit"
```

```bash
git branch -M main
```

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

```bash
git push -u origin main
```

---

# Disclaimer

This project is designed as an **AI-assisted resume analysis and comparison tool**.

The generated match score and candidate comparison are based on the provided Job Description and information extracted from uploaded resumes. They should be treated as decision-support information and reviewed by a human rather than used as an automated employment decision.

---

# Author

**Joyal Joseph**

Interested in:

* Machine Learning
* Deep Learning
* Generative AI
* Large Language Models
* Retrieval-Augmented Generation
* Data Science
* AI Engineering

---

## License

This project is intended for educational, portfolio, and demonstration purposes.
