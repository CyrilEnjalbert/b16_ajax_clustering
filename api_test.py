import httpx
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

# resp = client.get("/list_methods")
resp = client.get("/test")

print(f"reponse:{resp}")