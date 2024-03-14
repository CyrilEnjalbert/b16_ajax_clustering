from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import uvicorn
import requests
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
# ------------------------------------------ Predict ----------------------------------------------------------------

import pickle

# Load the model from file
with open('model_kmeans', 'rb') as file:
    model = pickle.load(file)


# ------------------------------------------ Model Choice ----------------------------------------------------------------

# Example request model
class PredictionInput(BaseModel):
    model_name: str
    params:list[str]


# Instance du moteur de modèles Jinja2 pour la gestion des templates HTML
templates = Jinja2Templates(directory="templates")

app = FastAPI()


    

# Example function to make predictions
def predict():
    # Call your machine learning model here and return predictions
    # Example: 
    prediction = model.predict()
    return prediction

@app.get('/choose_model')
async def run_prediction(request: Request):
    return templates.TemplateResponse("choose_model.html", {"request": request})

@app.post('/predict_model')
async def run_prediction_post(model_data: PredictionInput):
    try:
        model_name = model_data['model_name']
        prediction = predict()
        print(f"Prediction : {prediction} effectued with {model_name}")
        return JSONResponse(content={"redirect_url": "/plot_results"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/plot_results')
async def plot_results(request: Request):
    return templates.TemplateResponse("plot_results.html", {"request": request})

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)