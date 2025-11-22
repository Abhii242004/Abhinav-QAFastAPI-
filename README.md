Monolithic QA Automation & RAG Service

This is a monolithic Python backend application built with FastAPI designed to streamline the Quality Assurance (QA) and test automation process. It combines Retrieval-Augmented Generation (RAG) for knowledge retrieval, AI-powered test case generation, and automatic Selenium script creation.

🌟 Features

Document Ingestion: Ingest raw text and HTML content into a local vector store (ChromaDB).

Knowledge Retrieval: Uses RAG to find relevant context from ingested documents based on a user query.

Structured Test Case Generation: Uses Gemini LLM and RAG context to generate structured, step-by-step test cases in JSON format.

Selenium Script Generation: Translates a structured test case into a runnable Python Selenium WebDriver script.

🚀 Prerequisites

To run this application, you need the following:

Python 3.8+

API Key: Access to the Gemini API (the code is set up to use the __api_key provided in the runtime environment).

Web Driver: A compatible web browser (like Chrome) and its corresponding WebDriver executable (e.g., chromedriver) installed for running the generated Selenium scripts.

🛠️ Installation

Clone the repository (if applicable) and navigate to the project directory.

Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Linux/macOS
venv\Scripts\activate     # On Windows


Install Dependencies: Install all required packages using the requirements.txt file (or the list we generated).

# Based on the packages identified:
pip install fastapi uvicorn pydantic beautifulsoup4 lxml requests chromadb sentence-transformers


⚙️ Running the Application

Ensure you are in the virtual environment.

Start the FastAPI server using Uvicorn:

uvicorn app:app --reload


The application will typically start at http://127.0.0.1:8000.

🌐 API Endpoints

The service exposes three main endpoints:

Endpoint

Method

Description

Payload

Response

/upload_docs

POST

Ingests documents into the ChromaDB vector store. This must be done before generating test cases.

{"docs": List[str], "html": str}

{"status": str, "chunks_added": int}

/generate_test_cases

POST

Generates a structured test case (JSON) based on the user's query and the RAG context.

{"query": str}

GeneratedTestCase (JSON schema)

/generate_selenium

POST

Converts a structured test case (output from the previous step) into a runnable Python Selenium script.

{"test_case": Dict}

{"script": str} (Python code)

🧪 Usage Flow Example

INGEST: Send your application's documentation or website HTML to /upload_docs.

QUERY: Send a request to /generate_test_cases with a natural language query like: "Generate a test case for logging in with invalid credentials."

AUTOMATE: Take the JSON output from step 2 and send it to the /generate_selenium endpoint to receive the executable Python script.

⚠️ Important Notes

ChromaDB: The vector database is initialized in the chroma_test_db directory.

LLM Model: The application relies on the gemini-2.5-flash-preview-09-2025 model for all generative tasks.

Embedding Model: SentenceTransformer("all-MiniLM-L6-v2") is used for generating document embeddings.
