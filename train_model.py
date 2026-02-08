import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

# Importing dataset
data = pd.read_csv("personality_dataset1.csv")

# Check null
data.isnull().sum()

# Handling categorical data in 0 and 1
data['Stage_fear'] = data['Stage_fear'].map({'No': 0, 'Yes': 1})
data['Drained_after_socializing'] = data['Drained_after_socializing'].map({'No': 0, 'Yes': 1})
data["Personality"] = data["Personality"].map({"Introvert": 0, "Extrovert": 1})

# Defining label and features
X = data.iloc[:, 0:7]
y = data.iloc[:, 7]

# Splitting train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

# Training model
model = RandomForestClassifier(n_estimators=300, random_state=0)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation Metrics
# 1. Accuracy
accuracy = accuracy_score(y_test, y_pred)

# 2. Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
cm_table = pd.DataFrame(conf_matrix,
    index=["Actual Introvert", "Actual Extrovert"],
    columns=["Predicted Introvert", "Predicted Extrovert"])

# 3. Classification report
classification = classification_report(y_test, y_pred, output_dict=True)
report_table = pd.DataFrame(classification).transpose()
print("Evaluation Metrics")
print("1. Accuracy:\t", accuracy)
print("2. Confusion Matrix\n", cm_table)
print("3. Classification Report\n", report_table)

# Feature importance
feature_importance = pd.DataFrame({
    "Input": X.columns,
    "Weight": model.feature_importances_
})

# Save the trained model
with open('personality_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✓ Model saved as 'personality_model.pkl'")

# Save model metrics and data for frontend
model_data = {
    'model': model,
    'accuracy': accuracy,
    'conf_matrix': conf_matrix,
    'classification_report': classification,
    'feature_importance': feature_importance,
    'feature_names': X.columns.tolist()
}

with open('model_data.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("✓ Model data saved as 'model_data.pkl'")
print("Training complete! You can now run the Streamlit app.")