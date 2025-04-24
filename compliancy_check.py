import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

class ContractViolationChecker:
    def __init__(self):
        self.model = None
        self.columns = None

    def train_model(self, csv_path):
        # Load the data
        df = pd.read_csv(csv_path)
        X = df.drop(columns=['violation'])
        y = df['violation']
        self.columns = X.columns  # This stores the column names for prediction use
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

        # Train the model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        # Evaluate model
        y_pred = self.model.predict(X_test)
        print("Model Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))

    def predict(self, input_features):
        if self.model is None:
            raise Exception("Model not loaded or trained.")
            # Convert input features to DataFrame with the correct column names
        df = pd.DataFrame([input_features], columns=self.columns)
        return self.model.predict(df)[0]


    def save_model(self, model_path='models/compliancy_check.joblib'):
        # Save the model
        joblib.dump(self.model, model_path)
        print("Model saved successfully!")

    def load_model(self, model_path='models/compliancy_check.joblib'):
        # Load the model
        self.model = joblib.load(model_path)
        print("Model loaded successfully!")
