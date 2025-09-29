from flask import Flask, request, jsonify, render_template, send_from_directory
from medical_ai_model import MedicalAIModel
import os
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the medical AI model
medical_ai = MedicalAIModel()

@app.route('/')
def index():
    """Serve the chatbot interface"""
    return render_template('index.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return send_from_directory('.', filename)

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages and return AI predictions"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()

        if not message:
            return jsonify({
                'error': 'Please provide your symptoms',
                'disease': 'Unknown',
                'confidence': 0,
                'severity': 'Unknown',
                'consultation_urgency': 'Unknown',
                'treatments': [],
                'medicines': [],
                'prevention': []
            })

        # Get AI prediction
        result = medical_ai.predict_consultation(message)

        logger.info(f"User input: {message}")
        logger.info(f"AI prediction: {result['disease']} (confidence: {result.get('confidence', 0):.3f})")

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({
            'error': f'Server error: {str(e)}',
            'disease': 'Unknown',
            'confidence': 0,
            'severity': 'Unknown',
            'consultation_urgency': 'Unknown',
            'treatments': [],
            'medicines': [],
            'prevention': []
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_trained': medical_ai.is_trained,
        'timestamp': '2025-09-28 07:00:00 IST'
    })

def initialize_model():
    """Initialize and train the medical AI model"""
    logger.info("Initializing Medical AI Model...")

    # Try to load existing model first
    if os.path.exists('medical_ai_model.pkl'):
        logger.info("Loading existing model...")
        if medical_ai.load_model('medical_ai_model.pkl'):
            logger.info("Model loaded successfully!")
            return True

    # If no existing model, train a new one
    logger.info("Training new model...")
    if os.path.exists('comprehensive_medical_dataset.csv'):
        if medical_ai.train_models('comprehensive_medical_dataset.csv'):
            logger.info("Saving trained model...")
            medical_ai.save_model('medical_ai_model.pkl')
            logger.info("Model initialized successfully!")
            return True

    logger.error("Failed to initialize model!")
    return False

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')

    # Move index.html to templates directory if it exists in root
    if os.path.exists('index.html'):
        import shutil
        shutil.move('index.html', 'templates/index.html')
        logger.info("Moved index.html to templates directory")

    # Initialize the model
    if initialize_model():
        logger.info("Starting Medical AI Chatbot server...")
        logger.info("Access the chatbot at: http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        logger.error("Failed to initialize model. Cannot start server.")
