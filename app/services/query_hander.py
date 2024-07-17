class QueryHandler:
    def __init__(self, language_models, query_classifier):
        self.language_models = language_models
        self.classifier = query_classifier

    def handle_query(self, query, context=None):
        context = context or query
        query_type = self.classifier.classify(context)
        response = self.language_models.generate_response(query, query_type)
        return response, query_type