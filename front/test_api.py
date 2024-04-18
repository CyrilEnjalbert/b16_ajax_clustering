import asyncio
import pytest
import httpx
import uvicorn
import fastapi
import jinja2



# Fixture to set up data for testing
@pytest.fixture
def api_url():
    return "http://cyrilb15ajaxacr-fastfront.francecentral.azurecontainer.io:8001/docs"



# Test connection to the specified URL
@pytest.mark.asyncio
async def test_api_connection(api_url):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(api_url)
            assert response.status_code == 200  # Ensure HTTP status code is 200 (OK)
        except httpx.HTTPError as exc:
            assert False, f"HTTP error occurred: {exc}"  # Fail the test if an HTTP error occurs
            
# Run pytest if executed as a script
if __name__ == "__main__":
    asyncio.run(pytest.main(["-v", "-s", "test_api.py"]))