# Potato Disease Classification

A machine learning system that classifies potato plant diseases from images into three categories: Early Blight, Late Blight, or Healthy.

## Project Components

- **API**: FastAPI backend that serves the trained model
- **Frontend**: React web application for uploading and analyzing images
- **Mobile App**: React Native application for mobile devices
- **Training**: Jupyter notebooks for model training and conversion

## Setup Instructions

### Python Environment Setup

1. Install Python dependencies:

```bash
pip install -r training/requirements.txt
pip install -r api/requirements.txt
```

2. For API with TensorFlow Serving (optional):
   - Copy `models.config.example` as `models.config` and update paths
   - Run TF Serve with Docker:
   ```bash
   docker run -t --rm -p 8501:8501 -v /path/to/potato-disease-classification:/potato-disease-classification tensorflow/serving --rest_api_port=8501 --model_config_file=/potato-disease-classification/models.config
   ```

### Frontend Setup

1. Install Node.js and NPM
2. Install dependencies:

```bash
cd frontend
npm install
```

3. Configure environment:
   - Copy `.env.example` as `.env`
   - Update API URL in `.env` file

4. Start the frontend:

```bash
cd frontend
npm start
```

### Mobile App Setup

1. Install dependencies:

```bash
cd mobile-app
yarn install
```

2. For iOS (Mac only):
```bash
cd ios && pod install && cd ../
```

3. Configure environment:
   - Copy `.env.example` as `.env`
   - Update API URL in `.env` file

4. Run the app:
```bash
# For Android
npm run android

# For iOS
npm run ios
```

## Running the Application

### API Server

```bash
cd api
uvicorn potato_classifier_api:app --reload --host 0.0.0.0
```

The API will be available at `http://localhost:8000`

### Web Interface

```bash
cd frontend
npm start
```

The web interface will be available at `http://localhost:3000`

## Model Training

The model is trained on a dataset of potato plant leaves with three classes:
- Healthy
- Early Blight
- Late Blight

The model architecture consists of several convolutional layers followed by max pooling layers and fully connected layers. It achieves 99% accuracy on the test set.

## Compatibility Notes

- For newer Node.js versions (≥17), you may need to set the legacy OpenSSL provider:
```bash
export NODE_OPTIONS=--openssl-legacy-provider
```

- For PowerShell:
```powershell
$env:NODE_OPTIONS="--openssl-legacy-provider"
```

## Troubleshooting

If you encounter issues with model loading, use the simplified API in demo mode by modifying `api/potato_classifier_api.py` to not load the model.

