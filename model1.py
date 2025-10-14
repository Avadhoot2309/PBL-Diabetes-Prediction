# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  

# Load dataset
df = pd.read_csv(r"C:\Users\Avadhoot\Documents\VU STUDY MATERIAL\PBL\Python Project\Diabetes Prediction Dataset(in).csv")

# Display first few rows
print(df.head())

# Check for missing values
print("Missing values in dataset:\n", df.isnull().sum())

# Encode target column if it’s categorical
if df['diabetes'].dtype == 'object':
    target_encoder = LabelEncoder()
    df['diabetes'] = target_encoder.fit_transform(df['diabetes'])

# Convert categorical columns into numeric using one-hot encoding
df = pd.get_dummies(df, columns=['gender', 'smoking_history'], drop_first=True)

# Split dataset into features and label
X = df.drop('diabetes', axis=1)
y = df['diabetes']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display dataset sizes
print(f"Training data size: {X_train.shape[0]} samples")
print(f"Testing data size: {X_test.shape[0]} samples")

# Create and train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on test data
y_pred = model.predict(X_test)

# Calculate model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on test data: {accuracy*100:.2f}%")

# Display detailed classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Generate and display confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Save the trained model
joblib.dump(model, r"C:\Users\Avadhoot\Documents\VU STUDY MATERIAL\PBL\Python Project\diabetes_rf_model.pkl")
print("Model saved successfully!")

# Create new example input for testing
new_data = pd.DataFrame({
    'age': [45],
    'hypertension': [0],
    'heart_disease': [0],
    'bmi': [28.5],
    'HbA1c_level': [6.3],
    'blood_glucose_level': [140],
    'gender_1': [1], 
    'smoking_history_formerly_smoked': [0],
    'smoking_history_smokes': [1]
})

# Match new data columns with training data columns
new_data = new_data.reindex(columns=X.columns, fill_value=0)

# Predict on new data
prediction = model.predict(new_data)
if prediction[0] == 1:
    print("The person is likely to have diabetes.")
else:
    print("The person is not likely to have diabetes.")

# Plot feature importance
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
feature_importance.sort_values(ascending=False).plot(kind='bar', color='skyblue')
plt.title('Feature Importance in Diabetes Prediction')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.show()
