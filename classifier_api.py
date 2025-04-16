# filepath: /Users/dipayanrana/Desktop/potato-disease-classification-new/api/potato_classifier_api.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "API is running in demo mode"}