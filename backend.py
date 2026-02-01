from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ml_logic import StoryPointEstimator
import uvicorn

app = FastAPI(title="Story Point Estimator API")

class StoryInput(BaseModel):
    text: str

@app.post("/predict")
def predict_story_points(input_data: StoryInput):
    estimator = StoryPointEstimator.get_instance()
    try:
        result = estimator.predict(input_data.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
