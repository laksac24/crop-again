import os
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for all origins (everyone can access the API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow any origin to access the API
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Define the input model for the request body
class ModelInput(BaseModel):
    N: int
    P: int
    K: int
    temperature: float
    humidity: float
    ph: float
    rainfall: float

# Load the model from a pickle file
model = pickle.load(open("model1.pkl", "rb"))

# Prediction endpoint
@app.post('/predict')
def prediction(input_param: ModelInput):
    # Convert input to dictionary
    input_data = input_param.dict()

    # Extract features for the model
    nitrogen = input_data['N']
    phosphorous = input_data['P']
    potassium = input_data['K']
    temp = input_data['temperature']
    humid = input_data['humidity']
    phv = input_data['ph']
    rain = input_data['rainfall']

    # Prepare input for model prediction
    input_list = [nitrogen, phosphorous, potassium, temp, humid, phv, rain]

    # Get the model's prediction
    prediction = model.predict([input_list])

    # Mapping of model prediction to crop names
    crop_map = {
        1: 'apple',
        2: 'banana',
        3: 'rice',
        4: 'pomegranate',
        5: 'pigeonpeas',
        6: 'papaya',
        7: 'orange',
        8: 'muskmelon',
        9: 'mungbean',
        10: 'mothbeans',
        11: 'mango',
        12: 'maize',
        13: 'lentil',
        14: 'kidneybeans',
        15: 'jute',
        16: 'grapes',
        17: 'cotton',
        18: 'coffee',
        19: 'coconut',
        20: 'chickpea',
        21: 'blackgram',
        22: 'watermelon'
    }

    # Get the predicted crop name
    predicted_crop = crop_map.get(prediction[0], "Unknown crop")

    # Return the prediction as a JSON response
    return JSONResponse(content={"predicted_crop": predicted_crop})

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Crop Recommendation API"}
