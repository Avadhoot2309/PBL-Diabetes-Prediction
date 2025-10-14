import pandas as pd
import joblib

# Load the saved model
model = joblib.load(r"C:\Users\Avadhoot\Documents\VU STUDY MATERIAL\PBL\Python Project\diabetes_rf_model.pkl")
print("Model loaded successfully!\n")

# Define multiple test cases
test_cases = pd.DataFrame([
    # No diabetes / low risk
    {'age': 30, 'hypertension': 0, 'heart_disease': 0, 'bmi': 22.0, 'HbA1c_level': 5.5,
     'blood_glucose_level': 90, 'gender_1': 0, 'smoking_history_formerly_smoked': 0, 'smoking_history_smokes': 0},
    
    # Mild risk
    {'age': 45, 'hypertension': 1, 'heart_disease': 0, 'bmi': 28.0, 'HbA1c_level': 6.2,
     'blood_glucose_level': 120, 'gender_1': 1, 'smoking_history_formerly_smoked': 1, 'smoking_history_smokes': 0},
    
    # Moderate risk
    {'age': 50, 'hypertension': 1, 'heart_disease': 1, 'bmi': 30.0, 'HbA1c_level': 6.8,
     'blood_glucose_level': 140, 'gender_1': 1, 'smoking_history_formerly_smoked': 0, 'smoking_history_smokes': 1},
    
    # Severe risk / likely diabetes
    {'age': 60, 'hypertension': 1, 'heart_disease': 1, 'bmi': 35.0, 'HbA1c_level': 8.0,
     'blood_glucose_level': 200, 'gender_1': 0, 'smoking_history_formerly_smoked': 0, 'smoking_history_smokes': 1},
])

# Reindex to ensure columns match training columns
training_columns = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level',
                    'blood_glucose_level', 'gender_1', 
                    'smoking_history_formerly_smoked', 'smoking_history_smokes']
test_cases = test_cases.reindex(columns=training_columns, fill_value=0)

# Predict for all test cases
predictions = model.predict(test_cases)

# Display results
for i, prediction in enumerate(predictions):
    if prediction == 1:
        print(f"Test Case {i+1}: Likely to have diabetes.")
    else:
        print(f"Test Case {i+1}: Not likely to have diabetes.")
