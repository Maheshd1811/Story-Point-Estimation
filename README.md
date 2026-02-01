# Agile Story Point Estimator

This project provides an AI-based estimation for Agile User Stories.

## Structure
- `ml_logic.py`: Contains the model loading and prediction logic.
- `backend.py`: A FastAPI backend that exposes the prediction logic via a REST API.
- `frontend.py`: A Streamlit frontend that consumes the backend API.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running
1. **Start the Backend**:
   Open a terminal and run:
   ```bash
   python backend.py
   ```
   The backend will start at `http://localhost:8000`.

2. **Start the Frontend**:
   Open a **new** terminal and run:
   ```bash
   streamlit run frontend.py
   ```
   The frontend will open in your browser.
