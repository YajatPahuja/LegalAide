import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/legal_aid_dataset.csv")

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

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train the model
model = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Evaluate and print accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")


# Function to take user input and predict eligibility
def predict():
    user_input = []
    print("Enter values for the following features:")

    for feature in X.columns:
        value = input(f"{feature}: ").strip()

        # Convert numerical values
        if feature in numerical_features:
            try:
                user_input.append(float(value))
            except ValueError:
                print(f"Invalid input for {feature}. Please enter a number.")
                return

        # Encode categorical values
        elif feature in categorical_features:
            if value in encoders[feature].classes_:
                user_input.append(encoders[feature].transform([value])[0])
            else:
                print(f"Invalid category for {feature}. Expected one of {list(encoders[feature].classes_)}")
                return

    # Predict eligibility
    user_df = pd.DataFrame([user_input], columns=X.columns)
    prediction = model.predict(user_df)
    print(f"Predicted Outcome: {'Eligible' if prediction[0] == 1 else 'Not Eligible'}")

predict()
