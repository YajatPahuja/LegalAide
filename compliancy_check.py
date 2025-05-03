import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


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

        
    def compare_models(self, csv_path):
        df = pd.read_csv(csv_path)
        X = df.drop(columns=['violation'])
        y = df['violation']
        self.columns = X.columns

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'SVM': SVC(),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Naive Bayes': GaussianNB()
        }

        results = []

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            results.append({'Model': name, 'Accuracy': acc, 'F1 Score': f1})

        results_df = pd.DataFrame(results)
        print(results_df)

        # Plotting
        results_df.set_index('Model')[['Accuracy', 'F1 Score']].plot(
        kind='bar', figsize=(10,6), color=['skyblue', 'lightgreen']
        )
        plt.title('Model Comparison')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

checker = ContractViolationChecker()
# checker.compare_models('data/contracts.csv')



from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

checker = ContractViolationChecker()

# Train the model first
checker.train_model('data/contracts.csv')  # <-- make sure this path is correct

# Now you can safely access the first estimator
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

estimator = checker.model.estimators_[0]

plt.figure(figsize=(20, 10))
plot_tree(estimator, 
          feature_names=checker.columns, 
          class_names=["Not Eligible", "Eligible"], 
          filled=True, 
          rounded=True)
plt.title("A Single Tree from Random Forest")
plt.show()

