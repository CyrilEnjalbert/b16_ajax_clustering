# Import necessary modules and functions from models.py
import pytest
from models import prediction_kmeans, prediction_agglo
import time
import asyncio


# Fixture to set up data for testing
@pytest.fixture





# Test prediction_kmeans function
@pytest.mark.asyncio
async def test_prediction_kmeans():
    n_clusters = 3
    response = await prediction_kmeans(n_clusters)  # Await the coroutine
    assert response['model_name'] == 'Kmeans'
    assert response['n_clusters'] == n_clusters
    
time.sleep(5)
# Test prediction_agglo function
@pytest.mark.asyncio
async def test_prediction_agglo():
    n_clusters = 3
    response = await prediction_agglo(n_clusters)  # Await the coroutine
    assert response['model_name'] == 'Agglomeration'
    assert response['n_clusters'] == n_clusters


if __name__ == "__main__":
    # Run asyncio tasks within pytest
    asyncio.run(pytest.main(["-v", "-s", "test_models.py"]))