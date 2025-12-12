from http import HTTPStatus
import pytest

from src.api import app, get_model, get_feature_extractor_loaded

@pytest.fixture
def client(): 
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_reload(client):
    model = get_model()
    feature_extractor = get_feature_extractor_loaded()

    response = client.post("/reload")
    assert response.status_code == HTTPStatus.OK

    new_model = get_model()
    new_feature_extractor = get_feature_extractor_loaded()

    assert model is not new_model # ensure the model has been reloaded by checking object identity
    assert feature_extractor is not new_feature_extractor
            
def test_health(client):
    response = client.get("/health")
    assert response.status_code == HTTPStatus.OK
    assert response.data.decode() == "OK"