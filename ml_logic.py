import os
import joblib
import logging
import re
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AGILE_POINTS = [1, 2, 3, 5, 8, 13]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "ml_artifacts", "model.joblib")
VECTORIZER_PATH = os.path.join(BASE_DIR, "ml_artifacts", "vectorizer.joblib")

def clean_text(text: str) -> str:
    
    if not text:
        return ""
    
    text = text.lower()
    
    text = re.sub(r'[^a-z\s]', '', text)
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_story(text: str) -> str:
    return clean_text(text)

class StoryPointEstimator:
    _instance = None

    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.load_model()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_model(self):
        if self.model and self.vectorizer:
            return
        
        logger.info(f"Loading model from {MODEL_PATH}...")
        try:
            if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
                raise FileNotFoundError(f"Artifacts not found at {MODEL_PATH} or {VECTORIZER_PATH}")
            
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
        if distance < 0.2:
            return "high"
        elif distance < 0.5:
            return "medium"
        else:
            return "low"

    def predict(self, text: str):
        if not self.model or not self.vectorizer:
            self.load_model()
            
        cleaned_text = preprocess_story(text)
        vectors = self.vectorizer.transform([cleaned_text])
        raw_pred = float(self.model.predict(vectors)[0])
        final_points = self._find_nearest_point(raw_pred)
        confidence = self._calculate_confidence(raw_pred, final_points)
        
        return {
            "predicted_story_points": final_points,
            "raw_prediction": round(raw_pred, 2),
            "confidence": confidence
        }



