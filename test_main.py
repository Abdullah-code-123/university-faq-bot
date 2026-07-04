from starlette.testclient import TestClient
from weekproject import app
from unittest.mock import AsyncMock


client = TestClient(app)
class MockMessage:
     content = """{"explanation":"Test",
    "relatedTopics":["a","b"],
    "difficulty":1}"""
class MockChoice:
        message = MockMessage()
class MockResponse:
        choices = [MockChoice()]

async def fake_call(*args,**kwargs):
        return MockResponse()

def test_inputQ(monkeypatch):
    monkeypatch.setattr("weekproject.client.chat.completions.create",fake_call)
    response = client.post("/explain",json={"topic":"explain the VFS  detail"})
    assert response.status_code==200
    data=response.json()
    assert "explanation" in data
    assert "relatedTopics" in data
    assert isinstance(data["difficulty"],int)

def test_constarints():
    response1 = client.post("/explain",json={"topic":"pretend to be,system prompt,act as"})
    assert response1.status_code==400
    assert response1.json()["detail"]=="suspicious prompt detected"

def test_validation():
    response1 = client.post("/explain",json={"topic":"what"})
    assert response1.status_code==422
    assert "detail"in response1.json()

def test_rateLimit(monkeypatch):
    monkeypatch.setattr("weekproject.client.chat.completions.create",fake_call)
    for _ in range(11):
        response2=client.post("/explain",json={"topic":"Explain the most efficient algorithms for process sheduling in operating systems, including their time and space complexities, and provide examples of their applications in real-world scenarios."})
    assert response2.status_code==429
    assert "Too many requests try again after some time" in response2.json()["detail"]
