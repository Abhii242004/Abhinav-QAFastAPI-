# app.py - Monolithic Backend Application
# Combines Vector Store, Document Ingestion, Test Case Generation, and Selenium Scripting.

# --- 1. CORE IMPORTS AND CONFIGURATION ---
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator
from typing import List, Dict, Any, Optional
import hashlib
import time
import json
import os
import requests
from bs4 import BeautifulSoup
import chromadb
from sentence_transformers import SentenceTransformer

# LLM API Configuration
API_KEY = "" # Leave as-is; Canvas provides this at runtime.
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"

# Vector Store Configuration
DB_PATH = "chroma_test_db"
COLLECTION_NAME = "test_knowledge_base"


# --- 2. PYDANTIC SCHEMAS (DATA MODELS) ---

class UploadRequest(BaseModel):
    """Schema for the /upload_docs endpoint."""
    docs: List[str] # List of raw text lines or content items
    html: str      # Raw HTML string to be processed

class QueryRequest(BaseModel):
    """Schema for the /generate_test_cases endpoint."""
    query: str

class TestCaseRequest(BaseModel):
    """Schema for the /generate_selenium endpoint."""
    test_case: Dict # Expects a dictionary matching the structure of GeneratedTestCase

class TestStep(BaseModel):
    """Defines a single step in a test case for structured LLM output."""
    action: str  # e.g., "Click button", "Fill form", "Verify text"
    element: str # e.g., "Login button with text 'Sign In'", "Input field with name 'username'"
    value: Optional[str] = None # e.g., "testuser@example.com" if action is "Fill form"

class GeneratedTestCase(BaseModel):
    """The structured output schema for the TestCaseAgent."""
    test_case_name: str = Field(..., description="A short, descriptive name for the test case.")
    description: str = Field(..., description="A detailed explanation of the test case objective.")
    steps: List[TestStep] = Field(..., description="A sequence of detailed steps to execute the test.")


# --- 3. UTILITY FUNCTIONS ---

async def _call_llm_api(system_prompt: str, user_query: str, structured_schema: Optional[BaseModel] = None, context: Optional[str] = None) -> str:
    """
    Calls the Gemini API with exponential backoff and optional structured output.
    """
    if context:
        user_query = f"Context from Knowledge Base:\n---\n{context}\n---\n\nUser Request: {user_query}"
        
    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]}
    }

    headers = {'Content-Type': 'application/json'}
    
    if structured_schema:
        payload["generationConfig"] = {
            "responseMimeType": "application/json",
            "responseSchema": structured_schema.model_json_schema()
        }
    
    # Exponential backoff logic
    max_retries = 3
    delay = 1.0
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{API_URL}?key={API_KEY}",
                headers=headers,
                data=json.dumps(payload)
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Extract content from the result structure
            if result.get('candidates') and result['candidates'][0].get('content'):
                text = result['candidates'][0]['content']['parts'][0]['text']
                return text

            raise ValueError("LLM response was successful but content part is missing.")

        except (requests.exceptions.RequestException, ValueError) as e:
            if attempt < max_retries - 1 and response.status_code in [429, 500, 503]:
                time.sleep(delay)
                delay *= 2  # Exponential backoff
                continue
            
            print(f"LLM API Error: {e} | Last Response: {response.text}")
            raise HTTPException(status_code=500, detail=f"LLM API call failed: {e}")

    raise HTTPException(status_code=500, detail="LLM API call failed after multiple retries.")


# --- 4. CORE CLASSES ---

class VectorStore:
    """Manages ChromaDB operations: setup, embedding, chunking, and querying."""
    def __init__(self):
        try:
            self.client = chromadb.PersistentClient(path=DB_PATH) 
            self.collection = self.client.get_or_create_collection(COLLECTION_NAME)
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            self.chunk_size = 500
        except Exception as e:
            print(f"Error initializing VectorStore: {e}")
            raise

    def _get_embedding(self, text: str) -> List[float]:
        """Generates embeddings for a given text."""
        return self.embedder.encode(text).tolist()

    def _generate_id(self, source: str, content: str) -> str:
        """Generates a consistent ID."""
        data = f"{source}-{content}"
        return hashlib.sha256(data.encode()).hexdigest()

    def add_document(self, text: str, source: str) -> int:
        """Splits text into chunks and adds them to the database."""
        if not text.strip():
            return 0

        # Simple chunking logic
        chunks = [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        
        ids, embeddings, metadatas, documents = [], [], [], []
        
        for i, chunk in enumerate(chunks):
            ids.append(self._generate_id(source, chunk))
            embeddings.append(self._get_embedding(chunk))
            metadatas.append({"source": source, "chunk_index": i})
            documents.append(chunk)

        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        return len(chunks)

    def get_context(self, query: str, n_results: int = 5) -> str:
        """Retrieves and concatenates relevant text chunks based on a query."""
        if not self.collection.count():
            return "No documents available in the knowledge base."
            
        query_embedding = self._get_embedding(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['documents']
        )
        
        context_chunks = results['documents'][0] if results.get('documents') and results['documents'][0] else []
        return "\n---\n".join(context_chunks)

class DocumentIngestor:
    """Handles the processing of raw documents and passing them to the VectorStore."""
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def _clean_html(self, html_content: str, source: str) -> str:
        """Uses BeautifulSoup to extract clean text from HTML."""
        if not html_content.strip():
            return ""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            # Remove scripts, styles, and navigational elements
            for script_or_style in soup(['script', 'style', 'header', 'footer', 'nav']):
                script_or_style.decompose()
            
            text = soup.get_text()
            # Clean up excessive whitespace
            clean_text = ' '.join(text.split()).strip()
            return clean_text
        except Exception as e:
            print(f"Error cleaning HTML from source {source}: {e}")
            return ""

    def process_documents(self, docs: List[str], html: str):
        """Processes the list of raw text items and the HTML string."""
        total_chunks = 0
        
        # 1. Process HTML Content
        cleaned_html_text = self._clean_html(html, source="HTML_Input")
        if cleaned_html_text:
            total_chunks += self.vector_store.add_document(cleaned_html_text, source="HTML_Input")
        
        # 2. Process list of raw text documents/lines
        for i, doc_content in enumerate(docs):
            if doc_content.strip():
                source_id = f"Text_Doc_{i}"
                total_chunks += self.vector_store.add_document(doc_content, source=source_id)
        
        return total_chunks

class TestCaseAgent:
    """Agent responsible for generating structured test cases using RAG and LLM."""
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.system_prompt = (
            "You are an expert QA Engineer. Your task is to generate one complete, "
            "detailed functional test case based on the user's query and the provided context. "
            "You MUST use the exact JSON schema provided for your response. "
            "Ensure the element descriptions in the steps are specific enough for a QA automation tool."
        )

    async def generate_cases(self, query: str) -> Dict:
        """Retrieves context and calls LLM for structured test case generation."""
        context = self.vector_store.get_context(query, n_results=10)
        
        llm_response_text = await _call_llm_api(
            system_prompt=self.system_prompt,
            user_query=query,
            structured_schema=GeneratedTestCase,
            context=context
        )
        
        # The LLM response is already a JSON string matching the schema
        try:
            return json.loads(llm_response_text)
        except json.JSONDecodeError as e:
            print(f"Failed to parse LLM JSON response: {e}, Raw text: {llm_response_text}")
            raise HTTPException(status_code=500, detail="Failed to generate structured test case from LLM.")

class SeleniumScriptAgent:
    """Agent responsible for translating a structured test case into a Python Selenium script."""
    def __init__(self, vector_store: VectorStore):
        # We include the vector_store primarily to match the original structure,
        # but the agent operates without RAG context for this specific task.
        self.vector_store = vector_store
        self.system_prompt = (
            "You are an expert in Python and Selenium WebDriver. "
            "Your task is to convert the provided structured test case (JSON dictionary) into a complete, "
            "runnable Python script using the Selenium library. "
            "The script must include setup/teardown using a WebDriver (assume Chrome is installed), "
            "and use appropriate selectors (like By.XPATH, By.ID, or By.NAME) based on the element descriptions."
            "DO NOT include any explanation or commentary, ONLY output the Python code."
        )

    async def generate_script(self, test_case: Dict) -> str:
        """Calls LLM to generate the Selenium script based on the test case dictionary."""
        
        # We pass the structured test case directly as the user query
        user_query = f"Convert the following structured test case into a Python Selenium script:\n{json.dumps(test_case, indent=2)}"
        
        selenium_script = await _call_llm_api(
            system_prompt=self.system_prompt,
            user_query=user_query,
            structured_schema=None, # Expect raw text output (the code)
            context=None
        )
        
        # Clean up any markdown code fences the LLM might include
        if selenium_script.startswith("```python"):
            selenium_script = selenium_script.strip("```python").strip("`")
        
        return selenium_script


# --- 5. FASTAPI APPLICATION & ROUTES ---
# Initialize the application and core components
app = FastAPI(
    title="Monolithic QA Automation & RAG Service",
    description="A unified backend for data ingestion and AI-powered test case/script generation."
)
vector_store = VectorStore()
ingestor = DocumentIngestor(vector_store)
test_case_agent = TestCaseAgent(vector_store)
selenium_agent = SeleniumScriptAgent(vector_store)


@app.post("/upload_docs")
async def upload_docs(payload: UploadRequest):
    """
    Ingests documents (raw text list and HTML content) into the vector store.
    """
    try:
        chunks_added = ingestor.process_documents(payload.docs, payload.html)
        return {"status": "Knowledge Base Built", "chunks_added": chunks_added}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed during document ingestion: {e}")


@app.post("/generate_test_cases")
async def generate_test_cases(payload: QueryRequest):
    """
    Generates a structured test case using RAG/LLM based on the query.
    """
    if not vector_store.collection.count():
        raise HTTPException(status_code=404, detail="Knowledge base is empty. Please upload documents first via /upload_docs.")
        
    return await test_case_agent.generate_cases(payload.query)


@app.post("/generate_selenium")
async def generate_selenium(payload: TestCaseRequest):
    """
    Generates a Python Selenium script from a structured test case dictionary.
    """
    script_content = await selenium_agent.generate_script(payload.test_case)
    return {"script": script_content}