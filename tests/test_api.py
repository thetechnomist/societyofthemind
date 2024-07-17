import json
from app import create_app

def test_query_endpoint():
    app = create_app()
    client = app.test_client()
    
    response = client.post('/query', json={
        'query': 'Write a Python function to calculate fibonacci numbers',
        'conversation_id': 'test_conversation'
    })
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'response' in data
    assert 'classified_as' in data
    assert data['classified_as'] in ["general", "code", "creative"]