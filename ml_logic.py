import os
import joblib
import logging
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AGILE_POINTS = [1, 2, 3, 5, 8, 13]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "issues.csv")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "ml_artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.joblib")
VECTORIZER_PATH = os.path.join(ARTIFACTS_DIR, "vectorizer.joblib")

def clean_text(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class StoryPointEstimator:
    _instance = None

    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.ensure_artifacts_exist()
        self.load_model()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def ensure_artifacts_exist(self):
        if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
            logger.info("Artifacts not found. Initiating training process...")
            self.train_model()

    def train_model(self):
        if not os.path.exists(DATA_FILE):
            raise FileNotFoundError(f"Data file not found at {DATA_FILE}")

        logger.info(f"Loading data from {DATA_FILE}...")
        try:
            df = pd.read_csv(DATA_FILE, on_bad_lines='skip')
        except Exception as e:
            logger.error(f"Failed to read CSV: {e}")
            raise e

        df = df.dropna(subset=['title', 'storypoints'])
        df['description'] = df['description'].fillna('')
        df['text'] = df['title'] + " " + df['description']
        df['cleaned_text'] = df['text'].apply(clean_text)
        
        df['storypoints'] = pd.to_numeric(df['storypoints'], errors='coerce')
        df = df.dropna(subset=['storypoints'])
        df = df[df['storypoints'] <= 100]

        logger.info(f"Training on {len(df)} rows...")

        self.vectorizer = TfidfVectorizer(max_features=5000)
        X = self.vectorizer.fit_transform(df['cleaned_text'])
        y = df['storypoints']

        self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.model.fit(X, y)

        if not os.path.exists(ARTIFACTS_DIR):
            os.makedirs(ARTIFACTS_DIR)
        
        joblib.dump(self.model, MODEL_PATH)
        joblib.dump(self.vectorizer, VECTORIZER_PATH)
        logger.info(f"Model trained and saved to {ARTIFACTS_DIR}")

    def load_model(self):
        if self.model and self.vectorizer:
            return
        
        logger.info(f"Loading model from {MODEL_PATH}...")
        try:
            self.model = joblib.load(MODEL_PATH)
            self.vectorizer = joblib.load(VECTORIZER_PATH)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e

    def _find_nearest_point(self, value: float) -> int:
        if value < AGILE_POINTS[0]:
            return AGILE_POINTS[0]
        
        idx = (np.abs(np.array(AGILE_POINTS) - value)).argmin()
        return int(AGILE_POINTS[idx])

    def _calculate_confidence(self, raw_pred: float, mapped_point: int) -> str:
        distance = abs(raw_pred - mapped_point)
        if distance < 0.5:
            return "high"
        elif distance < 1.0:
            return "medium"
        else:
            return "low"

    def predict(self, text: str):
        if not self.model or not self.vectorizer:
            self.load_model()
            
        cleaned_text = clean_text(text)
        vectors = self.vectorizer.transform([cleaned_text])
        raw_pred = float(self.model.predict(vectors)[0])
        final_points = self._find_nearest_point(raw_pred)
        confidence = self._calculate_confidence(raw_pred, final_points)
        
        return {
            "predicted_story_points": final_points,
            "raw_prediction": round(raw_pred, 2),
            "confidence": confidence
        }
