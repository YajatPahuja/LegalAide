import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

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
    df[col] = le.fit_transform(df[col])  # Encode the categorical column
    encoders[col] = le  # Store encoder for later use

# Define features and target
X = df.drop(columns=["Eligibility Status"])
y = df["Eligibility Status"]

# Train Decision Tree Model
model = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42)
model.fit(X, y)

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

# Run prediction function
predict()
