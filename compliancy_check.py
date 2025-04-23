import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


df = pd.read_csv("data/contracts.csv")

X = df.drop(columns=['violation'])
y = df['violation']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


y_pred = rf_model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

new_contract = pd.DataFrame([[20, 100, 1, 5]], columns=X.columns)

prediction = rf_model.predict(new_contract)
print("Prediction for new contract:", "Violation" if prediction[0] == 1 else "Compliant")

# # Save the trained model
# joblib.dump(rf_model, "rf_model.pkl")
# print("Model saved successfully!")
