from flask import Flask
from app.config import Config
from app.api.routes import api_bp
from app.models.language_models import LanguageModels
from app.services.query_hander import QueryHandler
from app.services.context_manager import ContextManager
from app.models.classifier import QueryClassifier
from flask_redis import FlaskRedis
import logging
import torch


import torch



def create_app(config_class=Config):
    app = Flask(__name__)
    redis_client = FlaskRedis()

    app.config.from_object(config_class)

    # Initialize Flask-Redis with the app
    redis_client.init_app(app)

    # Register blueprints
    app.register_blueprint(api_bp)

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Log device information
    if torch.backends.mps.is_available():
        device = 'Apple Silicon GPU (MPS)'
    elif torch.cuda.is_available():
        device = 'NVIDIA GPU (CUDA)'
    else:
        device = 'CPU'
    app.logger.info(f"Using {device} for computations")
    # Initialize services
    language_models = LanguageModels()
    query_classifier = QueryClassifier()
    context_manager = ContextManager(redis_client)
    query_handler = QueryHandler(language_models, query_classifier)

    # Fit the classifier with some initial data
    # You should replace this with actual training data
    initial_X = ["Python code", "Creative writing", "General knowledge"]
    initial_y = ["code", "creative", "general"]
    query_classifier.fit(initial_X, initial_y)

        # Make services available to the entire app
    app.language_models = language_models
    app.query_classifier = query_classifier
    app.context_manager = context_manager
    app.query_handler = query_handler

    return app
