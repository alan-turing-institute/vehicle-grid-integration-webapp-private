from fastapi.testclient import TestClient
from vgi_api import app

client = TestClient(app)


def test_simulation():
    response = client.post("/simulate")
    assert response.status_code == 200
