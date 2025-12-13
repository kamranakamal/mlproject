import os
import sys
import pandas as pd
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import joblib
from src.exception import CustomException
from src.logger import logging


app = FastAPI()

PREPROCESSOR_PATH = os.path.join("artifacts", "preprocessor.pkl")
MODEL_PATH = os.path.join("artifacts", "model.pkl")

try:
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)
except Exception as exc:
    logging.error("Failed to load artifacts", exc_info=exc)
    raise CustomException(exc, sys)


class InputData(BaseModel):
    gender: str
    race_ethnicity: str
    parental_level_of_education: str
    lunch: str
    test_preparation_course: str
    reading_score: int
    writing_score: int


@app.post("/predict")
def predict(data: InputData):
    try:
        payload = {
            "gender": data.gender,
            "race/ethnicity": data.race_ethnicity,
            "parental level of education": data.parental_level_of_education,
            "lunch": data.lunch,
            "test preparation course": data.test_preparation_course,
            "reading score": data.reading_score,
            "writing score": data.writing_score,
        }
        input_df = pd.DataFrame([payload])
        features = preprocessor.transform(input_df)
        prediction = model.predict(features)
        return {"prediction": int(prediction[0])}
    except Exception as exc:
        logging.error("Prediction failed", exc_info=exc)
        raise CustomException(exc, sys)


if __name__ == "__main__":
    uvicorn.run(app)