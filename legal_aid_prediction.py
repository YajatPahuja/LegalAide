import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , f1_score
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt

class LegalAidModel:
    def __init__(self, model_path='models/legal_aid_model.joblib', encoder_path='models/encoders.joblib'):
        self.model = None
        self.encoders = {}
        self.model_path = model_path
        self.encoder_path = encoder_path
        self.columns = None

    def train_model(self, csv_path):
        # Load dataset
        df = pd.read_csv(csv_path)

        # Identify categorical and numerical columns
        categorical_features = [
            "Gender", "Education Level", "Marital Status", "Employment Status", 
            "Legal Issue", "Urgency Level", "Prior Legal History", 
            "Disability Status", "Citizenship Status", "Criminal Record"
        ]
        numerical_features = ["Age", "Annual Income", "Number of Dependents"]

        # Encode categorical features
        encoders = {}
        for col in categorical_features:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

        # Define features and target
        X = df.drop(columns=["Eligibility Status"])
        y = df["Eligibility Status"]

        # Capture column order from the trained model
        self.columns = X.columns.tolist()

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # Train the model
        model = DecisionTreeClassifier(criterion="gini", max_depth=50, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate and print accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.2f}")

        # Save the model and encoders
        joblib.dump(model, self.model_path)
        joblib.dump(encoders, self.encoder_path)
        print("Model and encoders saved successfully!")

    def load_model(self, model_path='model/ipc_knn_model.joblib'):
        model = joblib.load(model_path)
        self.vectorizer = model['vectorizer']
        self.model = model['model']
        self.merged_data = model['merged_data']
        self.features = self.vectorizer.transform(self.merged_data['Combined_Text'])

    def predict(self, user_input):
        if self.model is None or not self.encoders:
            raise Exception("Model not loaded.")

        # Convert input into a DataFrame
        user_data = []
        categorical_features = [
            "Gender", "Education Level", "Marital Status", "Employment Status", 
            "Legal Issue", "Urgency Level", "Prior Legal History", 
            "Disability Status", "Citizenship Status", "Criminal Record"
        ]
        numerical_features = ["Age", "Annual Income", "Number of Dependents"]

        for feature in categorical_features + numerical_features:  # Ensure both categorical and numerical features are included
            value = user_input.get(feature)
            if feature in self.encoders:
                if value in self.encoders[feature].classes_:
                    user_data.append(self.encoders[feature].transform([value])[0])
                else:
                    raise ValueError(f"Invalid category for {feature}. Expected one of {list(self.encoders[feature].classes_)}")
            else:
                user_data.append(value)

        # Ensure that the columns passed match the model's expectation (including both categorical and numerical features)
        user_df = pd.DataFrame([user_data], columns=self.columns)  # Use the columns from training
        prediction = self.model.predict(user_df)
        return 'Eligible' if prediction[0] == 1 else 'Not Eligible'
        
    def compare_models(self, csv_path):
       # Load and preprocess dataset
        df = pd.read_csv(csv_path)

        categorical_features = [
               "Gender", "Education Level", "Marital Status", "Employment Status", 
                "Legal Issue", "Urgency Level", "Prior Legal History", 
               "Disability Status", "Citizenship Status", "Criminal Record"
        ]
        numerical_features = ["Age", "Annual Income", "Number of Dependents"]

        for col in numerical_features:
           if df[col].isnull().any():
              df[col].fillna(df[col].median(), inplace=True)
    
        for col in categorical_features:
           if df[col].isnull().any():
              df[col].fillna(df[col].mode()[0], inplace=True)

        # Encode categorical features
        encoders = {}
        for col in categorical_features:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

        X = df.drop(columns=["Eligibility Status"])
        y = df["Eligibility Status"]
        self.columns = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
           X, y, test_size=0.3, random_state=42, stratify=y
        )

    # Models to compare
        models = {
            'Decision Tree': DecisionTreeClassifier(max_depth=50, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Naive Bayes': GaussianNB(),
            'SVM': SVC()
        }

        results = []

        for name, model in models.items():
           model.fit(X_train, y_train)
           y_pred = model.predict(X_test)
           acc = accuracy_score(y_test, y_pred)
           f1 = f1_score(y_test, y_pred, average='binary')  # Use 'binary' for 0/1 labels
           results.append({'Model': name, 'Accuracy': acc, 'F1 Score': f1})

        results_df = pd.DataFrame(results)
        print(results_df)

        # Plotting accuracy
        ax = results_df.set_index('Model')[['Accuracy', 'F1 Score']].plot(
        kind='bar', figsize=(10, 6), color=['skyblue', 'lightgreen']
    )
        plt.title("Model Comparison: Accuracy vs F1 Score")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.show()



# Fixed user input for testing
user_input = {
    "Age": 30,
    "Gender": "Male",
    "Education Level": "High School",
    "Marital Status": "Married",
    "Annual Income": 45000,
    "Employment Status": "Unemployed",
    "Number of Dependents": 2,
    "Legal Issue": "Criminal",
    "Urgency Level": "High",
    "Prior Legal History": "No",
    "Disability Status": "No",
    "Citizenship Status": "Citizen",
    "Criminal Record": "No"
}


# # Initialize model, load it and make a prediction
# legal = LegalAidModel()
# legal.load_model()

# # Predict eligibility
# result = legal.predict(user_input)
# print(f"Predicted Outcome: {result}")

legal = LegalAidModel()
# legal.compare_models("data/legal_aid_dataset.csv")


# import pandas as pd
# import joblib
# from sklearn.tree import plot_tree
# import matplotlib.pyplot as plt
# from sklearn.tree import DecisionTreeClassifier

# # Load the trained model
# model = joblib.load('models/legal_aid_model.joblib')

# # Ensure model is a decision tree
# if not isinstance(model, DecisionTreeClassifier):
#     raise TypeError("Loaded model is not a DecisionTreeClassifier. Only decision trees can be visualized using plot_tree.")

# # Load the dataset to get feature names
# df = pd.read_csv('data/legal_aid_dataset.csv')
# X = df.drop(columns=["Eligibility Status"])

# # Plot the tree with improvements
# plt.figure(figsize=(24, 12))  # Wider figure
# plot_tree(model, 
#           feature_names=X.columns, 
#           class_names=["Not Eligible", "Eligible"], 
#           filled=True, 
#           rounded=True, 
#           max_depth=4,        # Limit depth to avoid clutter
#           fontsize=10,        # Smaller font for readability
#           impurity=False)     # Optional: hide impurity values for cleaner plot

# plt.title("Legal Aid Eligibility - Decision Tree (Depth Limited)", fontsize=16)
# plt.tight_layout()
# plt.show()
