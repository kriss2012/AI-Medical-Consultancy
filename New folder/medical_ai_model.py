import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
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
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

class MedicalAIModel:
    def __init__(self):
        self.disease_model = None
        self.treatment_model = None
        self.medicine_model = None
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        self.label_encoders = {}
        self.is_trained = False

    def preprocess_text(self, text):
        """Preprocess text for better feature extraction"""
        if pd.isna(text):
            return ""

        # Convert to lowercase
        text = str(text).lower()

        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove stopwords
        try:
            stop_words = set(stopwords.words('english'))
            words = word_tokenize(text)
            text = ' '.join([word for word in words if word not in stop_words])
        except:
            pass

        return text

    def load_data(self, csv_file='comprehensive_medical_dataset.csv'):
        """Load and preprocess the medical dataset"""
        try:
            df = pd.read_csv(csv_file)
            print(f"Loaded dataset with {len(df)} records")

            # Preprocess symptoms text
            df['symptoms_clean'] = df['symptoms'].apply(self.preprocess_text)

            # Remove any empty records
            df = df.dropna(subset=['symptoms_clean', 'disease'])
            df = df[df['symptoms_clean'].str.len() > 0]

            print(f"After preprocessing: {len(df)} records")
            return df

        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def prepare_features_and_labels(self, df):
        """Prepare features and labels for training"""
        # Features (symptoms)
        X = df['symptoms_clean'].values

        # Labels
        y_disease = df['disease'].values
        y_treatment = df['treatment'].values
        y_medicine = df['medicines'].values
        y_severity = df['severity'].values
        y_consultation = df['consultation_needed'].values

        # Encode categorical labels
        encoders = {}

        # Disease encoder
        disease_encoder = LabelEncoder()
        y_disease_encoded = disease_encoder.fit_transform(y_disease)
        encoders['disease'] = disease_encoder

        # Severity encoder
        severity_encoder = LabelEncoder()
        y_severity_encoded = severity_encoder.fit_transform(y_severity)
        encoders['severity'] = severity_encoder

        # Consultation encoder
        consultation_encoder = LabelEncoder()
        y_consultation_encoded = consultation_encoder.fit_transform(y_consultation)
        encoders['consultation'] = consultation_encoder

        self.label_encoders = encoders

        return X, {
            'disease': y_disease_encoded,
            'treatment': y_treatment,
            'medicine': y_medicine,
            'severity': y_severity_encoded,
            'consultation': y_consultation_encoded,
            'disease_names': y_disease,
            'treatment_names': y_treatment,
            'medicine_names': y_medicine
        }

    def train_models(self, csv_file='comprehensive_medical_dataset.csv'):
        """Train the AI models"""
        print("Loading and preprocessing data...")
        df = self.load_data(csv_file)

        if df is None:
            print("Failed to load data")
            return False

        # Prepare features and labels
        X, labels = self.prepare_features_and_labels(df)

        # Vectorize features
        print("Vectorizing features...")
        X_vectorized = self.tfidf_vectorizer.fit_transform(X)

        # Split data
        X_train, X_test, indices_train, indices_test = train_test_split(
            X_vectorized, range(len(X)), test_size=0.2, random_state=42, stratify=labels['disease']
        )

        # Get corresponding labels for train/test sets
        y_train_disease = labels['disease'][indices_train]
        y_test_disease = labels['disease'][indices_test]

        y_train_severity = labels['severity'][indices_train]
        y_test_severity = labels['severity'][indices_test]

        y_train_consultation = labels['consultation'][indices_train]
        y_test_consultation = labels['consultation'][indices_test]

        print("Training disease prediction model...")
        # Train disease prediction model
        disease_model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            class_weight='balanced'
        )
        disease_model.fit(X_train, y_train_disease)

        # Evaluate disease model
        y_pred_disease = disease_model.predict(X_test)
        disease_accuracy = accuracy_score(y_test_disease, y_pred_disease)
        print(f"Disease prediction accuracy: {disease_accuracy:.3f}")

        print("Training severity prediction model...")
        # Train severity prediction model
        severity_model = GradientBoostingClassifier(random_state=42)
        severity_model.fit(X_train, y_train_severity)

        # Evaluate severity model
        y_pred_severity = severity_model.predict(X_test)
        severity_accuracy = accuracy_score(y_test_severity, y_pred_severity)
        print(f"Severity prediction accuracy: {severity_accuracy:.3f}")

        print("Training consultation prediction model...")
        # Train consultation urgency model
        consultation_model = LogisticRegression(random_state=42, max_iter=1000)
        consultation_model.fit(X_train, y_train_consultation)

        # Evaluate consultation model
        y_pred_consultation = consultation_model.predict(X_test)
        consultation_accuracy = accuracy_score(y_test_consultation, y_pred_consultation)
        print(f"Consultation urgency accuracy: {consultation_accuracy:.3f}")

        # Store models and additional data
        self.disease_model = disease_model
        self.severity_model = severity_model
        self.consultation_model = consultation_model

        # Store treatment and medicine mappings
        self.treatment_mapping = dict(zip(df['disease'], df['treatment']))
        self.medicine_mapping = dict(zip(df['disease'], df['medicines']))
        self.prevention_mapping = dict(zip(df['disease'], df['prevention']))

        self.is_trained = True

        # Print detailed classification report for disease prediction
        print("\nDisease Prediction - Classification Report:")
        disease_names = self.label_encoders['disease'].classes_
        print(classification_report(y_test_disease, y_pred_disease, 
                                  target_names=disease_names, zero_division=0))

        return True

    def predict_consultation(self, symptoms):
        """Predict disease, treatment, and medicine recommendations"""
        if not self.is_trained:
            return {
                'error': 'Model not trained. Please train the model first.',
                'disease': 'Unknown',
                'confidence': 0,
                'severity': 'Unknown',
                'consultation_urgency': 'Unknown',
                'treatments': [],
                'medicines': [],
                'prevention': []
            }

        try:
            # Preprocess input
            symptoms_clean = self.preprocess_text(symptoms)

            if not symptoms_clean:
                return {
                    'error': 'Please provide valid symptoms',
                    'disease': 'Unknown',
                    'confidence': 0,
                    'severity': 'Unknown',
                    'consultation_urgency': 'Unknown',
                    'treatments': [],
                    'medicines': [],
                    'prevention': []
                }

            # Vectorize input
            symptoms_vectorized = self.tfidf_vectorizer.transform([symptoms_clean])

            # Predict disease
            disease_pred = self.disease_model.predict(symptoms_vectorized)[0]
            disease_proba = self.disease_model.predict_proba(symptoms_vectorized)[0]
            disease_name = self.label_encoders['disease'].inverse_transform([disease_pred])[0]
            confidence = max(disease_proba)

            # Predict severity
            severity_pred = self.severity_model.predict(symptoms_vectorized)[0]
            severity_name = self.label_encoders['severity'].inverse_transform([severity_pred])[0]

            # Predict consultation urgency
            consultation_pred = self.consultation_model.predict(symptoms_vectorized)[0]
            consultation_name = self.label_encoders['consultation'].inverse_transform([consultation_pred])[0]

            # Get treatments and medicines
            treatments = self.treatment_mapping.get(disease_name, '').split(' | ')
            medicines = self.medicine_mapping.get(disease_name, '').split(' | ')
            preventions = self.prevention_mapping.get(disease_name, '').split(' | ')

            return {
                'disease': disease_name,
                'confidence': float(confidence),
                'severity': severity_name,
                'consultation_urgency': consultation_name,
                'treatments': [t.strip() for t in treatments if t.strip()],
                'medicines': [m.strip() for m in medicines if m.strip()],
                'prevention': [p.strip() for p in preventions if p.strip()],
                'recommendation': self.generate_recommendation(disease_name, severity_name, consultation_name)
            }

        except Exception as e:
            return {
                'error': f'Prediction error: {str(e)}',
                'disease': 'Unknown',
                'confidence': 0,
                'severity': 'Unknown',
                'consultation_urgency': 'Unknown',
                'treatments': [],
                'medicines': [],
                'prevention': []
            }

    def generate_recommendation(self, disease, severity, urgency):
        """Generate personalized recommendation"""
        base_msg = f"Based on your symptoms, you may have {disease} with {severity.lower()} severity."

        if urgency == 'Urgent':
            urgency_msg = " Please seek immediate medical attention."
        elif urgency == 'Within 24 hours':
            urgency_msg = " Please consult a healthcare provider within 24 hours."
        elif urgency == 'Within a week':
            urgency_msg = " Please schedule an appointment with your healthcare provider within a week."
        else:
            urgency_msg = " Schedule a routine check-up with your healthcare provider."

        disclaimer = " This is an AI-based assessment and should not replace professional medical advice."

        return base_msg + urgency_msg + disclaimer

    def save_model(self, filepath='medical_ai_model.pkl'):
        """Save the trained model"""
        if not self.is_trained:
            print("Model not trained yet!")
            return False

        try:
            model_data = {
                'disease_model': self.disease_model,
                'severity_model': self.severity_model,
                'consultation_model': self.consultation_model,
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'label_encoders': self.label_encoders,
                'treatment_mapping': self.treatment_mapping,
                'medicine_mapping': self.medicine_mapping,
                'prevention_mapping': self.prevention_mapping,
                'is_trained': self.is_trained
            }

            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)

            print(f"Model saved successfully to {filepath}")
            return True

        except Exception as e:
            print(f"Error saving model: {e}")
            return False

    def load_model(self, filepath='medical_ai_model.pkl'):
        """Load a pre-trained model"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            self.disease_model = model_data['disease_model']
            self.severity_model = model_data['severity_model']
            self.consultation_model = model_data['consultation_model']
            self.tfidf_vectorizer = model_data['tfidf_vectorizer']
            self.label_encoders = model_data['label_encoders']
            self.treatment_mapping = model_data['treatment_mapping']
            self.medicine_mapping = model_data['medicine_mapping']
            self.prevention_mapping = model_data['prevention_mapping']
            self.is_trained = model_data['is_trained']

            print(f"Model loaded successfully from {filepath}")
            return True

        except Exception as e:
            print(f"Error loading model: {e}")
            return False

def train_medical_ai():
    """Function to train and save the medical AI model"""
    print("Initializing Medical AI Model...")
    model = MedicalAIModel()

    print("Training models...")
    success = model.train_models('comprehensive_medical_dataset.csv')

    if success:
        print("Saving trained model...")
        model.save_model('medical_ai_model.pkl')
        print("Training completed successfully!")

        # Test the model with sample input
        print("\nTesting model with sample symptoms...")
        test_symptoms = "I have fever, headache, and body aches"
        result = model.predict_consultation(test_symptoms)

        print(f"\nTest Input: {test_symptoms}")
        print(f"Predicted Disease: {result['disease']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Severity: {result['severity']}")
        print(f"Consultation Urgency: {result['consultation_urgency']}")
        print(f"Recommendation: {result['recommendation']}")
    else:
        print("Training failed!")

    return model

if __name__ == '__main__':
    train_medical_ai()
