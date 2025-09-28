#!/usr/bin/env python3
"""
Medical AI Training and Testing Script
=====================================

This script trains the medical AI model and tests it with various scenarios.
Run this script to train your AI model before using the chatbot application.

Usage:
    python train_and_test.py
"""

import os
import sys
import pandas as pd
import numpy as np
from medical_ai_model import MedicalAIModel, train_medical_ai
import warnings
warnings.filterwarnings('ignore')

def test_model_predictions():
    """Test the trained model with various symptom scenarios"""
    print("\n" + "="*60)
    print("TESTING MEDICAL AI MODEL")
    print("="*60)

    # Initialize model
    model = MedicalAIModel()

    # Load the trained model
    if not model.load_model('medical_ai_model.pkl'):
        print("Error: Could not load trained model!")
        return False

    # Test cases with expected outcomes
    test_cases = [
        {
            'symptoms': 'I have a high fever, chills, body aches, and fatigue',
            'description': 'Flu-like symptoms'
        },
        {
            'symptoms': 'I am experiencing chest pain, shortness of breath, and irregular heartbeat',
            'description': 'Heart-related symptoms'
        },
        {
            'symptoms': 'I have a persistent cough, fever, and difficulty breathing',
            'description': 'Respiratory symptoms'
        },
        {
            'symptoms': 'I feel dizzy, have blurred vision, and frequent urination',
            'description': 'Diabetes-like symptoms'
        },
        {
            'symptoms': 'I have severe headache, nausea, and sensitivity to light',
            'description': 'Migraine symptoms'
        },
        {
            'symptoms': 'I have stomach pain, nausea, vomiting, and loss of appetite',
            'description': 'Gastric symptoms'
        },
        {
            'symptoms': 'I have joint pain, stiffness, and swelling in my hands',
            'description': 'Joint-related symptoms'
        },
        {
            'symptoms': 'I feel sad, have lost interest in activities, and trouble sleeping',
            'description': 'Mental health symptoms'
        }
    ]

    print(f"\nTesting {len(test_cases)} different symptom scenarios...\n")

    for i, test_case in enumerate(test_cases, 1):
        print(f"TEST {i}: {test_case['description']}")
        print(f"Input: '{test_case['symptoms']}'")
        print("-" * 40)

        # Get prediction
        result = model.predict_consultation(test_case['symptoms'])

        # Display results
        print(f"🔍 Predicted Disease: {result['disease']}")
        print(f"📊 Confidence: {result['confidence']:.1%}")
        print(f"⚡ Severity: {result['severity']}")
        print(f"⏰ Consultation: {result['consultation_urgency']}")

        if result.get('treatments'):
            print(f"🏥 Treatments: {', '.join(result['treatments'][:3])}")

        if result.get('medicines'):
            print(f"💊 Medicines: {', '.join(result['medicines'][:3])}")

        print(f"📋 Recommendation: {result.get('recommendation', 'N/A')}")
        print("\n" + "="*60 + "\n")

    return True

def validate_dataset():
    """Validate the medical dataset"""
    print("\n" + "="*60)
    print("VALIDATING MEDICAL DATASET")
    print("="*60)

    if not os.path.exists('comprehensive_medical_dataset.csv'):
        print("Error: comprehensive_medical_dataset.csv not found!")
        return False

    df = pd.read_csv('comprehensive_medical_dataset.csv')

    print(f"✅ Dataset loaded successfully")
    print(f"📊 Total records: {len(df)}")
    print(f"🏥 Unique diseases: {df['disease'].nunique()}")
    print(f"📋 Columns: {list(df.columns)}")

    # Check for missing values
    missing_values = df.isnull().sum()
    print(f"\n🔍 Missing values per column:")
    for col, missing in missing_values.items():
        if missing > 0:
            print(f"   {col}: {missing}")
        else:
            print(f"   {col}: ✅ No missing values")

    # Show disease distribution
    print(f"\n📈 Top 10 most common diseases in dataset:")
    top_diseases = df['disease'].value_counts().head(10)
    for disease, count in top_diseases.items():
        print(f"   {disease}: {count} records")

    # Show severity distribution
    print(f"\n⚡ Severity distribution:")
    severity_dist = df['severity'].value_counts()
    for severity, count in severity_dist.items():
        print(f"   {severity}: {count} records")

    # Show consultation urgency distribution
    print(f"\n⏰ Consultation urgency distribution:")
    urgency_dist = df['consultation_needed'].value_counts()
    for urgency, count in urgency_dist.items():
        print(f"   {urgency}: {count} records")

    return True

def main():
    """Main function to run training and testing"""
    print("="*60)
    print("MEDICAL AI TRAINING AND TESTING SYSTEM")
    print("="*60)
    print("This script will:")
    print("1. Validate the medical dataset")
    print("2. Train the AI model")
    print("3. Test the trained model with sample cases")
    print("4. Save the trained model for use in the chatbot")
    print("="*60)

    # Step 1: Validate dataset
    if not validate_dataset():
        print("❌ Dataset validation failed!")
        sys.exit(1)

    # Step 2: Train the model
    print("\n" + "="*60)
    print("TRAINING MEDICAL AI MODEL")
    print("="*60)

    try:
        model = train_medical_ai()
        print("✅ Model training completed successfully!")
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        sys.exit(1)

    # Step 3: Test the model
    if not test_model_predictions():
        print("❌ Model testing failed!")
        sys.exit(1)

    # Final success message
    print("="*60)
    print("🎉 SUCCESS! Medical AI system is ready to use!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run the chatbot: python medical_chatbot_app.py")
    print("2. Open your browser to: http://localhost:5000")
    print("3. Start chatting with your Medical AI assistant!")
    print("\n" + "="*60)

if __name__ == '__main__':
    main()
