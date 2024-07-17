from locust import HttpUser, task, between

class APIUser(HttpUser):
    wait_time = between(1, 5)
    
    @task
    def query_api(self):
        self.client.post("/query", json={
            "query": "What is the capital of France?",
            "conversation_id": "performance_test"
        })