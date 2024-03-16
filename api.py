from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import uvicorn
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse
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

class PredictionInput(BaseModel):
    model_name: str
    params:list[str]

# @app.get('/choose_model_test')
# async def run_prediction(model_name: str):
#     if model_name == "Kmeans":
#         image_path = "plot_kmeans.png"
#         return FileResponse(image_path, media_type="image/png")
#     else:
#         return HTTPException(status_code=404, detail="Model not found")

@app.get('/test') 
async def test():
    print("coucou")

@app.post("/choose_model/{model_name}")
async def run_prediction(n_clusters: int, model_name = str):
    if model_name == "Kmeans":
    # Send the data to the receiver API
        async with httpx.AsyncClient() as client:
            response = await client.post("http://localhost:8001/prediction_kmeans", params={"n_clusters": n_clusters})

            return response.text

    if model_name == "agglo":
    # Send the data to the receiver API
        async with httpx.AsyncClient() as client:
            response = await client.post("http://localhost:8001/prediction_agglo", params={"n_clusters": n_clusters})
            
            return response.text
        
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)