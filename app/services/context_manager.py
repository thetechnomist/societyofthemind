from flask_redis import FlaskRedis

class ContextManager:
    def __init__(self, redis_client: FlaskRedis):
        self.redis_client = redis_client

    def get_context(self, conversation_id):
        if conversation_id is None:
            return None
        try:
            history = self.redis_client.get(conversation_id)
            return history.decode('utf-8') if history else None
        except Exception as e:
            print(f"Error retrieving context: {str(e)}")
            return None

    def update_context(self, conversation_id, context, response):
        if conversation_id is None:
            return
        try:
            updated_context = f"{context}\n{response}"
            self.redis_client.set(conversation_id, updated_context)
        except Exception as e:
            print(f"Error updating context: {str(e)}")