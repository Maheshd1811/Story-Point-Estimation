# Interview Guide: Agile User Story Point Estimator

## 1. Project Overview
**One-Liner:** "I built an end-to-end Machine Learning application that automates the estimation of Agile user stories using Natural Language Processing."

**The Problem:** Agile teams often struggle with inconsistent story point estimates. One developer might say "3 points" while another says "8 points" for the same task. This variation leads to unpredictable sprint planning.

**The Solution:** This tool provides an objective, AI-driven baseline. It analyzes the text of a user story and predicts its complexity coverage based on historical data, mapping it to standard Agile Fibonacci numbers (1, 2, 3, 5, 8, 13).

---

## 2. Technical Architecture & Stack

### **Frontend: Streamlit**
*   **Why:** It allows for rapid development of data-centric web apps completely in Python.
*   **Role:** Acts as the client. It captures user input, sends requests to the backend, and renders the specific "Blue/Slate" professional theme and usage instructions.

### **Backend: FastAPI**
*   **Why:** It's one of the fastest Python web frameworks, supports asynchronous operations, and auto-generates API documentation.
*   **Role:** Exposes a REST API (`/predict`). It decouples the ML logic from the UI, meaning the model could theoretically be used by a Slack bot, a Jira plugin, or a mobile app in the future.

### **Machine Learning: Scikit-Learn**
*   **Algorithm:** **Random Forest Regressor**.
    *   *Why?* Random Forests are robust, handle non-linear relationships well, and are less prone to overfitting than a single Decision Tree.
*   **Text Processing:** **TF-IDF (Term Frequency-Inverse Document Frequency) Vectorizer**.
    *   *Why?* It converts raw text into numerical vectors that the model can understand, highlighting unique/important words while downplaying common ones (like "the", "a").
*   **Logic:**
    1.  **Preprocessing:** Clean text (lowercase, remove punctuation).
    2.  **Vectorization:** Transform text to vectors.
    3.  **Inference:** Model predicts a continuous number (e.g., `4.2`).
    4.  **Post-processing:** The system maps that raw number to the nearest valid Story Point (e.g., `4.2` becomes `5`).

---

## 3. How to Explain the "Flow" in an Interview

1.  **"Data Ingestion":** "First, the user inputs a story like 'As a user, I want to login...' into the Streamlit UI."
2.  **"API Call":** "The frontend sends a JSON payload to the FastAPI backend."
3.  **"Feature Engineering":** "The backend passes the text to the logic layer, where we effectively 'clean' the text using Regex and then vectorize it using a pre-trained TF-IDF vectorizer."
4.  **"Inference":** "The vector hits the Random Forest model, which outputs a raw complexity score."
5.  **"Business Logic mapping":** "Since Agile uses specific numbers (1, 2, 3, 5, 8...), we calculate the nearest valid neighbor. For example, a prediction of 6.8 is snapped to 8."
6.  **"Response":** "The final result and a confidence score are returned to the user."

---

## 4. Potential Interview Questions

**Q1: Why did you choose Random Forest over Deep Learning (e.g., BERT/Transformers)?**
*   **Answer:** "For this specific problem size and complexity, Random Forest combined with TF-IDF provides a highly efficient and interpretable baseline. Transformers like BERT are computationally expensive and might be overkill (and slower) for simple complexity estimation tasks unless we have a massive dataset."

**Q2: How do you handle new words that weren't in the training set?**
*   **Answer:** "The TF-IDF vectorizer ignores words it hasn't seen before. If a story is full of completely new jargon, the model relies on the known words. To improve this, I would retrain the vectorizer periodically with new data."

**Q3: Is this a Classification or Regression problem?**
*   **Answer:** "It is modeled as a **Regression** problem because complexity is continuous. However, strictly speaking, the output is categorical (ordinal) because we map it to fixed buckets (1, 2, 3, 5...). Using regression allows the model to interpolate—e.g., understanding that a story is 'between' a 3 and a 5—before we make the final decision."

**Q4: How would you deploy this to production?**
*   **Answer:** "I would containerize the application using **Docker** (one container for frontend, one for backend). I'd orchestrate them with **Docker Compose** or **Kubernetes**. I would serve the FastAPI app using a production server like Gunicorn with Uvicorn workers."

**Q5: What was the biggest challenge?**
*   **Answer:** "One challenge was handling the disconnect between raw model outputs (e.g., 3.7) and rigid Agile points. I implemented a post-processing logic to find the nearest valid neighbor, which ensures the tool speaks the same language as the developers."
