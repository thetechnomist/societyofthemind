from flask import Blueprint, request, jsonify, current_app
import traceback

api_bp = Blueprint('api', __name__)

@api_bp.route('/query', methods=['POST'])
def handle_query():
    try:
        data = request.json
        if data is None:
            return jsonify({"error": "No JSON data received"}), 400
        
        query = data.get('query')
        if query is None:
            return jsonify({"error": "No query provided in the JSON data"}), 400
        
        conversation_id = data.get('conversation_id')
        
        # Get context (will be None if conversation_id is None)
        context = current_app.context_manager.get_context(conversation_id)
        
        # If there's no existing context, use the query as the context
        context = context or query
        
        response, query_type = current_app.query_handler.handle_query(query, context)
        
        # Update context only if conversation_id is provided
        if conversation_id:
            current_app.context_manager.update_context(conversation_id, context, response)
        
        return jsonify({
            'response': response,
            'classified_as': query_type,
            'conversation_id': conversation_id
        })
    except Exception as e:
        current_app.logger.error(f"Error processing query: {str(e)}")
        current_app.logger.error(traceback.format_exc())
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

@api_bp.errorhandler(500)
def internal_server_error(e):
    current_app.logger.error(f"Unhandled exception: {str(e)}")
    current_app.logger.error(traceback.format_exc())
    return jsonify({"error": "Internal server error", "details": str(e)}), 500