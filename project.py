from ipc_recommendation import IPCRecommender
from compliancy_check import ContractViolationChecker
from legal_aid_prediction import LegalAidModel

# recommender = IPCRecommender()
# recommender.load_model('models/ipc_model.joblib')

# # Now use it directly
# case_input = "The accused forcibly entered a house and committed theft at night."
# recommendations = recommender.recommend(case_input)

# # Print or return recommendations
# for section, desc, score in recommendations:
#     print(f"Section {section}")
#     print(f"Description: {desc.strip()}")
#     print(f"Similarity Score: {score:.3f}")
#     print("-" * 50)

# checker = ContractViolationChecker();

# checker.load_model();

# new_input = [20, 100, 1, 5]
# prediction = checker.predict(new_input)

# print("Prediction:", "Violation" if prediction == 1 else "Compliant")

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

# Initialize model, load it and make a prediction
legal = LegalAidModel()
legal.load_model()

# Predict eligibility
result = legal.predict(user_input)
print(f"Predicted Outcome: {result}")
