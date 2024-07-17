import pytest
from app.models.language_models import LanguageModels

def test_language_models_initialization():
    lm = LanguageModels()
    assert hasattr(lm, 'general_lm')
    assert hasattr(lm, 'code_lm')
    assert hasattr(lm, 'creative_lm')

def test_generate_response():
    lm = LanguageModels()
    response = lm.generate_response("Hello, world!", "general")
    assert isinstance(response, str)
    assert len(response) > 0

@pytest.mark.parametrize("query_type", ["general", "code", "creative"])
def test_generate_response_for_all_types(query_type):
    lm = LanguageModels()
    response = lm.generate_response(f"Test query for {query_type}", query_type)
    assert isinstance(response, str)
    assert len(response) > 0