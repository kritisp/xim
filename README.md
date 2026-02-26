# Title Verification System

A fast, scalable API to verify titles based on predefined rules, phonetic similarity, and semantic uniqueness.

## Project Structure

- `model-service/`: A dedicated FastAPI app that hosts the ML model (SentenceTransformers).
- `backend/`: The main FastAPI app managing rules and coordinating similarity checks.
- `frontend/`: Simple HTML/JS interface to test the API.
- `data/`: Sample datasets and rules configurations.

## Running the Application

### 1. Start the Model Service (Runs on port 8001 by default, or 8000 depending on config)
```bash
cd model-service
pip install -r requirements.txt
uvicorn main:app --port 8001 --reload
```

### 2. Start the Backend API (Runs on port 8080)
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --port 8080 --reload
```

### 3. Test the Frontend
Open `frontend/index.html` in your browser.
