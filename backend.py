import os

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ml_logic import StoryPointEstimator


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="Story Point Estimator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StoryInput(BaseModel):
    text: str


@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))


@app.get("/style.css")
def serve_styles():
    return FileResponse(os.path.join(BASE_DIR, "style.css"), media_type="text/css")


@app.get("/script.js")
def serve_script():
    return FileResponse(
        os.path.join(BASE_DIR, "script.js"),
        media_type="application/javascript",
    )


@app.post("/predict")
def predict_story_points(input_data: StoryInput):
    try:
        estimator = StoryPointEstimator.get_instance()
        result = estimator.predict(input_data.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
