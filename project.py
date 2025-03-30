import streamlit as st
st.title("LEGAL AID")

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the datasets
ipc_data = pd.read_csv("data/ipc_sections.csv")
fir_data = pd.read_csv("data/FIR_DATASET.csv")

# Fill missing values with empty strings
ipc_data.fillna('', inplace=True)
fir_data.fillna('', inplace=True)

# Merge both datasets on common 'Description' (or adjust based on actual structure)
merged_data = ipc_data.copy()
if 'Cognizable' in fir_data.columns and 'Bailable' in fir_data.columns and 'Court' in fir_data.columns:
    merged_data = merged_data.merge(fir_data[['Description', 'Cognizable', 'Bailable', 'Court']], 
                                    on='Description', how='left')

# Fill any remaining missing values
merged_data.fillna('', inplace=True)

# Combine relevant text fields for similarity checking
merged_data['Combined_Text'] = (merged_data['Description'] + ' ' + 
                                merged_data['Offense'] + ' ' + 
                                merged_data['Punishment'] + ' ' + 
                                merged_data.get('Cognizable', '') + ' ' + 
                                merged_data.get('Bailable', '') + ' ' + 
                                merged_data.get('Court', ''))

# Use CountVectorizer for text representation
count_vectorizer = CountVectorizer(stop_words='english')
count_matrix = count_vectorizer.fit_transform(merged_data['Combined_Text'])

# Function to find most relevant IPC sections using Cosine Similarity
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

# Example test case
if __name__ == "__main__":
    test_case = st.text_input("Enter Your name", "Type Here ...")
    top_matches = recommend_ipc_sections_cosine(test_case)

    print("\nTop Recommended IPC Sections (Cosine Similarity):\n")
    for section, description, similarity in top_matches:
        st.header(f"Section{section}")
        st.write(f"Description: {description.strip()}\nSimilarity Score: {similarity:.3f}\n")


# name = st.text_input("Enter Your name", "Type Here ...")

# # display the name when the submit button is clicked
# # .title() is used to get the input text string
# if(st.button('Submit')):
#     result = name.title()
#     st.success(result)