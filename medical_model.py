import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass

def load_medical_data():
    """Load and preprocess the medical dataset"""
    try:
        df = pd.read_csv('comprehensive_medical_dataset.csv')
        print(f"Loaded dataset with {len(df)} records")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def preprocess_text(text):
    """Clean and preprocess text data"""
    if pd.isna(text):
        return ""

    # Convert to lowercase
    text = str(text).lower()

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    # Remove stopwords
    try:
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text)
        text = ' '.join([word for word in words if word not in stop_words])
    except:
        pass

    return text

def extract_features(df):
    """Extract features from the dataset"""
    # Combine symptoms with other relevant text features
    df['combined_text'] = (df['symptoms'].astype(str) + ' ' + 
                          df['severity'].astype(str) + ' ' + 
                          df['age_group'].astype(str))

    # Preprocess text
    df['processed_text'] = df['combined_text'].apply(preprocess_text)

    return df

def train_medical_model():
    """Train the medical diagnosis model"""
    print("Loading medical data...")
    df = load_medical_data()
    if df is None:
        return None, None, None

    print("Extracting features...")
    df = extract_features(df)

    # Prepare features and labels
    X_text = df['processed_text'].values
    y = df['disease'].values

    # Create TF-IDF vectorizer
    print("Creating TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8,
        stop_words='english'
    )

    X = vectorizer.fit_transform(X_text)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training models...")

    # Train multiple models
    models = {
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=42)
    }

    best_model = None
    best_score = 0
    best_model_name = ""

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)

        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        avg_score = np.mean(cv_scores)

        print(f"{name} - CV Accuracy: {avg_score:.4f} (+/- {np.std(cv_scores) * 2:.4f})")

        if avg_score > best_score:
            best_score = avg_score
            best_model = model
            best_model_name = name

    print(f"\nBest model: {best_model_name} with accuracy: {best_score:.4f}")

    # Test the best model
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Save the model and vectorizer
    model_data = {
        'model': best_model,
        'vectorizer': vectorizer,
        'model_name': best_model_name,
        'accuracy': test_accuracy,
        'diseases': list(df['disease'].unique()),
        'dataset': df  # Include dataset for additional information
    }

    with open('medical_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)

    print("Model saved successfully!")
    return best_model, vectorizer, df

def load_medical_model():
    """Load the trained medical model"""
    try:
        with open('medical_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        print("Model not found. Training new model...")
        model, vectorizer, df = train_medical_model()
        return load_medical_model()

def get_disease_info(disease, dataset):
    """Get detailed information about a disease"""
    disease_data = dataset[dataset['disease'] == disease]
    if len(disease_data) == 0:
        return None

    # Get the most common information for this disease
    info = {
        'disease': disease,
        'common_treatments': list(set([t for treatments in disease_data['treatment'] 
                                     for t in treatments.split(' | ')]))[:5],
        'common_medicines': list(set([m for medicines in disease_data['medicines'] 
                                    for m in medicines.split(' | ')]))[:5],
        'prevention': list(set([p for prevention in disease_data['prevention'] 
                               for p in prevention.split(' | ')]))[:5],
        'severity': disease_data['severity'].mode().iloc[0] if len(disease_data['severity'].mode()) > 0 else 'Unknown',
        'specialist': disease_data['specialist_required'].mode().iloc[0] if len(disease_data['specialist_required'].mode()) > 0 else 'General Practitioner',
        'consultation_urgency': disease_data['consultation_needed'].mode().iloc[0] if len(disease_data['consultation_needed'].mode()) > 0 else 'routine',
        'average_confidence': disease_data['confidence_score'].mean()
    }

    return info

def medical_diagnosis(symptoms_text):
    """Diagnose based on symptoms"""
    try:
        # Load the model
        model_data = load_medical_model()
        model = model_data['model']
        vectorizer = model_data['vectorizer']
        dataset = model_data['dataset']

        # Preprocess the input
        processed_symptoms = preprocess_text(symptoms_text)

        # Vectorize the input
        X_input = vectorizer.transform([processed_symptoms])

        # Get prediction probabilities
        probabilities = model.predict_proba(X_input)[0]
        predicted_disease = model.predict(X_input)[0]
        confidence = max(probabilities)

        # Get top 3 predictions
        top_indices = np.argsort(probabilities)[-3:][::-1]
        top_diseases = [(model.classes_[i], probabilities[i]) for i in top_indices]

        # Get detailed information for the predicted disease
        disease_info = get_disease_info(predicted_disease, dataset)

        result = {
            'primary_diagnosis': predicted_disease,
            'confidence': float(confidence * 100),
            'top_predictions': [{'disease': disease, 'probability': float(prob * 100)} 
                              for disease, prob in top_diseases],
            'disease_info': disease_info,
            'recommendation': generate_recommendation(disease_info, confidence)
        }

        return result

    except Exception as e:
        return {
            'error': f'Error in diagnosis: {str(e)}',
            'primary_diagnosis': 'Unknown',
            'confidence': 0,
            'recommendation': 'Please consult a healthcare professional.'
        }

def generate_recommendation(disease_info, confidence):
    """Generate medical recommendations based on diagnosis"""
    if disease_info is None:
        return "Please consult a healthcare professional for proper diagnosis."

    recommendations = []

    # Add urgency-based recommendation
    urgency = disease_info.get('consultation_urgency', 'routine')
    if urgency == 'urgent':
        recommendations.append("⚠️ URGENT: Seek immediate medical attention.")
    elif urgency == 'within24hours':
        recommendations.append("🏥 Consult a doctor within 24 hours.")
    else:
        recommendations.append("👨‍⚕️ Schedule a routine consultation with your doctor.")

    # Add specialist recommendation
    specialist = disease_info.get('specialist', 'General Practitioner')
    if specialist != 'General Practitioner':
        recommendations.append(f"🔬 Consider consulting a {specialist}.")

    # Add confidence-based recommendation
    if confidence < 0.7:
        recommendations.append("⚠️ Low confidence in diagnosis. Please seek professional medical evaluation.")

    # Add general advice
    recommendations.append("📋 This AI diagnosis is for informational purposes only and should not replace professional medical advice.")

    return recommendations

if __name__ == "__main__":
    # Train the model
    print("Training medical diagnosis model...")
    train_medical_model()
    print("Training completed!")
