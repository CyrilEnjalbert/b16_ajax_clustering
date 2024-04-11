from fastapi import FastAPI
import uvicorn
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import httpx


# Instance du moteur de modèles Jinja2 pour la gestion des templates HTML
templates = Jinja2Templates(directory="templates")

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.post("/choose_model/{model_name}")
async def run_prediction(n_clusters: int, model_name = str):
    if model_name == "Kmeans":
    # Send the data to the receiver API
        async with httpx.AsyncClient() as client:
            response = await client.post("http://172.21.0.3:8001/prediction_kmeans", params={"n_clusters": n_clusters})

            return response.text

    if model_name == "agglo":
    # Send the data to the receiver API
        async with httpx.AsyncClient() as client:
            response = await client.post("http://172.21.0.3:8001/prediction_agglo", params={"n_clusters": n_clusters})
            
            return response.text
        
        
    
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)