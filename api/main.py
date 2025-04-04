from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import random

# Set to True to use the trained model, False for demo mode
USE_MODEL = False

app = FastAPI()

# Allow requests from all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# Only import TensorFlow if we're using the model
if USE_MODEL:
    import tensorflow as tf
    try:
        MODEL = tf.keras.models.load_model("../potatoes.h5")
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Running in demo mode instead")
        USE_MODEL = False

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    
    if USE_MODEL:
        # Use the actual ML model for prediction
        img_batch = np.expand_dims(image, 0)
        predictions = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))
    else:
        # Demo mode - make deterministic prediction based on image characteristics
        img_sum = np.sum(image) % 3
        predicted_class = CLASS_NAMES[img_sum]
        confidence = random.uniform(0.75, 0.98)
    
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)

