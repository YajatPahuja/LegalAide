import streamlit as st

# Import your model classes
from ipc_recommendation import IPCRecommenderKNN
# from compliancy_check import ContractViolationChecker
# from legal_aid_prediction import LegalAidModel

st.set_page_config(page_title="Legal AI Hub", layout="centered")

st.title("🔍 Legal AI Assistant")
page = st.sidebar.selectbox("Choose a service", ["IPC Section Recommender", "Contract Compliance", "Legal Aid Prediction"])

# IPC RECOMMENDER
if page == "IPC Section Recommender":
    st.header("📖 IPC Section Recommender")

    description = st.text_area("Enter FIR/Case Description:")
    if st.button("Get Recommendations"):
        with st.spinner("Loading model and predicting..."):
            recommender = IPCRecommenderKNN()
            recommender.load_model('models/ipc_knn_model.joblib')
            recommendations = recommender.recommend(description)
            st.success("Top Matches:")
            for section, desc, score in recommendations:
                st.markdown(f"**Section {section}**\n\n_{desc}_\n\n**Score**: {score:.2f}")
                st.markdown("---")

# CONTRACT COMPLIANCE
elif page == "Contract Compliance":
    st.header("📄 Contract Violation Checker")

    st.markdown("**(Placeholder UI)** – add contract input fields here.")
    # Example usage:
    # new_input = [20, 100, 1, 5]
    # checker = ContractViolationChecker()
    # checker.load_model()
    # prediction = checker.predict(new_input)
    # st.write("Prediction:", "Violation" if prediction == 1 else "Compliant")

# LEGAL AID PREDICTION
elif page == "Legal Aid Prediction":
    st.header("⚖️ Legal Aid Eligibility Predictor")

    st.markdown("**(Placeholder UI)** – add form fields for user input.")
    # Example input:
    # user_input = {
    #     "Age": 30,
    #     "Gender": "Male",
    #     ...
    # }
    # legal = LegalAidModel()
    # legal.load_model()
    # result = legal.predict(user_input)
    # st.write(f"Predicted Outcome: {result}")
