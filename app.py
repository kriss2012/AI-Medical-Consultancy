from flask import Flask, request, jsonify, render_template, send_from_directory
from medical_model import medical_diagnosis, load_medical_model, train_medical_model
import os
import json
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Global variables
model_data = None
consultation_history = []

def initialize_model():
    """Initialize the medical model"""
    global model_data
    try:
        logger.info("Loading medical model...")
        model_data = load_medical_model()
        logger.info("Medical model loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.info("Training new model...")
        try:
            train_medical_model()
            model_data = load_medical_model()
            logger.info("New model trained and loaded successfully!")
            return True
        except Exception as train_error:
            logger.error(f"Error training model: {train_error}")
            return False

@app.route('/')
def index():
    """Serve the main application"""
    return send_from_directory('.', 'index.html')

@app.route('/api/diagnose', methods=['POST'])
def diagnose():
    """Main diagnosis endpoint"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                'error': 'No data provided',
                'success': False
            }), 400

        symptoms = data.get('symptoms', '').strip()
        if not symptoms:
            return jsonify({
                'error': 'Please provide symptoms',
                'success': False
            }), 400

        # Perform diagnosis
        logger.info(f"Diagnosing symptoms: {symptoms[:100]}...")
        result = medical_diagnosis(symptoms)

        # Add timestamp and save to history
        result['timestamp'] = datetime.now().isoformat()
        result['symptoms_input'] = symptoms
        result['success'] = True

        # Save to consultation history
        consultation_history.append(result)

        # Keep only last 100 consultations
        if len(consultation_history) > 100:
            consultation_history.pop(0)

        logger.info(f"Diagnosis completed: {result['primary_diagnosis']} ({result['confidence']:.1f}%)")

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in diagnosis: {e}")
        return jsonify({
            'error': f'Diagnosis failed: {str(e)}',
            'success': False,
            'primary_diagnosis': 'Error',
            'confidence': 0,
            'recommendation': ['Please try again or consult a healthcare professional.']
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    global model_data
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_data is not None,
        'timestamp': datetime.now().isoformat(),
        'total_consultations': len(consultation_history)
    })

@app.route('/api/consultation-history', methods=['GET'])
def get_consultation_history():
    """Get consultation history"""
    limit = request.args.get('limit', 10, type=int)
    return jsonify({
        'consultations': consultation_history[-limit:],
        'total': len(consultation_history)
    })

@app.route('/api/diseases', methods=['GET'])
def get_diseases():
    """Get list of diseases the model can diagnose"""
    global model_data
    if model_data:
        return jsonify({
            'diseases': model_data.get('diseases', []),
            'model_accuracy': model_data.get('accuracy', 0),
            'model_name': model_data.get('model_name', 'Unknown')
        })
    else:
        return jsonify({
            'error': 'Model not loaded',
            'diseases': []
        }), 503

@app.route('/api/symptom-analysis', methods=['POST'])
def symptom_analysis():
    """Advanced symptom analysis"""
    try:
        data = request.get_json()
        symptoms = data.get('symptoms', '').strip()

        if not symptoms:
            return jsonify({'error': 'No symptoms provided'}), 400

        # Get diagnosis
        diagnosis_result = medical_diagnosis(symptoms)

        # Add additional analysis
        analysis = {
            'word_count': len(symptoms.split()),
            'character_count': len(symptoms),
            'symptoms_complexity': 'Simple' if len(symptoms.split()) < 10 else 'Detailed',
            'urgency_indicators': check_urgency_keywords(symptoms),
            'diagnosis': diagnosis_result
        }

        return jsonify(analysis)

    except Exception as e:
        logger.error(f"Error in symptom analysis: {e}")
        return jsonify({'error': str(e)}), 500

def check_urgency_keywords(symptoms):
    """Check for urgent symptoms keywords"""
    urgent_keywords = [
        'severe', 'intense', 'unbearable', 'emergency', 'acute',
        'sudden', 'rapid', 'difficulty breathing', 'chest pain',
        'high fever', 'bleeding', 'unconscious', 'seizure'
    ]

    symptoms_lower = symptoms.lower()
    found_keywords = [keyword for keyword in urgent_keywords 
                     if keyword in symptoms_lower]

    return {
        'urgent_keywords_found': found_keywords,
        'urgency_level': 'High' if found_keywords else 'Normal',
        'immediate_attention_needed': len(found_keywords) > 1
    }

@app.route('/api/retrain-model', methods=['POST'])
def retrain_model():
    """Retrain the model (admin function)"""
    try:
        logger.info("Retraining model...")
        train_medical_model()

        # Reload the model
        global model_data
        model_data = load_medical_model()

        return jsonify({
            'success': True,
            'message': 'Model retrained successfully',
            'model_accuracy': model_data.get('accuracy', 0),
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error retraining model: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🏥 Starting MediAI - Medical Consultation App...")
    print("📊 Initializing medical diagnosis model...")

    # Initialize the model
    if initialize_model():
        print("✅ Model loaded successfully!")
        print("🌐 Starting Flask server...")
        print("📱 Access the app at: http://localhost:5000")
        print("🔬 API Health Check: http://localhost:5000/api/health")
        print("💡 API Diagnose: http://localhost:5000/api/diagnose")

        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to initialize model. Please check your dataset.")
        print("📋 Make sure 'fake_review_dataset.csv' is in the current directory.")
