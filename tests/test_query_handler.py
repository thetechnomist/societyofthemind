import pytest
from unittest.mock import patch, Mock
from app.services.query_hander import QueryHandler
from app.services.context_manager import ContextManager
from app.models.language_models import LanguageModels
from app.models.classifier import QueryClassifier

@pytest.fixture
def setup_services():
    language_models = LanguageModels()
    query_classifier = QueryClassifier()
    context_manager = ContextManager(Mock())
    query_handler = QueryHandler(language_models, query_classifier)

        # Fit the classifier with some initial data
    # You should replace this with actual training data
    initial_X = ["Python code", "Creative writing", "General knowledge"]
    initial_y = ["code", "creative", "general"]
    query_classifier.fit(initial_X, initial_y)

    return query_handler, context_manager

def test_query_handler_integration(setup_services):
    query_handler, context_manager = setup_services

    query = "Write a Python function to calculate fibonacci numbers"
    conversation_id = "test_conversation"

    context = context_manager.get_context(conversation_id)
    context = f"{context}\n{query}" if context else query

    response, query_type = query_handler.handle_query(query, context)
    
    assert isinstance(response, str)
    assert query_type in ["general", "code", "creative"]

@patch('app.models.language_models.LanguageModels.generate_response')
def test_query_handler_with_mock(mock_generate_response, setup_services):
    mock_generate_response.return_value = "Mocked response"
    query_handler, _ = setup_services

    response, query_type = query_handler.handle_query("Test query", "Test context")
    
    assert response == "Mocked response"
    mock_generate_response.assert_called_once()