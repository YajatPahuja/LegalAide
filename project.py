import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("LEGAL AID - IPC Section Recommender")

# Load the datasets
ipc_data = pd.read_csv("data/ipc_sections.csv")
fir_data = pd.read_csv("data/FIR_DATASET.csv")

# Fill missing values with empty strings
ipc_data.fillna('', inplace=True)
fir_data.fillna('', inplace=True)

# Merge both datasets on common 'Description'
merged_data = ipc_data.copy()
if 'Cognizable' in fir_data.columns and 'Bailable' in fir_data.columns and 'Court' in fir_data.columns:
    merged_data = merged_data.merge(fir_data[['Description', 'Cognizable', 'Bailable', 'Court']], 
                                    on='Description', how='left')

merged_data.fillna('', inplace=True)

# Combine relevant text fields
merged_data['Combined_Text'] = (
    merged_data['Description'] + ' ' + 
    merged_data['Offense'] + ' ' + 
    merged_data['Punishment'] + ' ' + 
    merged_data.get('Cognizable', '') + ' ' + 
    merged_data.get('Bailable', '') + ' ' + 
    merged_data.get('Court', '')
)

# Vectorization
count_vectorizer = CountVectorizer(stop_words='english')
count_matrix = count_vectorizer.fit_transform(merged_data['Combined_Text'])

# Cosine similarity function
def recommend_ipc_sections_cosine(case_description, top_n=3):
    case_vector = count_vectorizer.transform([case_description])
    similarities = cosine_similarity(case_vector, count_matrix)[0]
    
    recommendations = []
    for index, similarity in enumerate(similarities):
        if similarity > 0.1:
            row = merged_data.iloc[index]
            recommendations.append((row['Section'], row['Description'], similarity))
    
    recommendations = sorted(recommendations, key=lambda x: x[2], reverse=True)[:top_n]
    return recommendations

# Streamlit form
case_input = st.text_area("Enter Case Description", "Type the details of your case here...")

if st.button("Get IPC Recommendations"):
    top_matches = recommend_ipc_sections_cosine(case_input)
    if top_matches:
        st.subheader("Top Recommended IPC Sections:")
        for section, description, similarity in top_matches:
            st.markdown(f"### Section {section}")
            st.write(f"**Description:** {description.strip()}")
            st.write(f"**Similarity Score:** {similarity:.3f}")
            st.markdown("---")
    else:
        st.warning("No relevant IPC sections found. Please enter more detailed case information.")
