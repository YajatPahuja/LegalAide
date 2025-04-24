import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

class IPCRecommender:
    def __init__(self, ipc_path='data/ipc_sections.csv', fir_path='data/FIR_DATASET.csv'):
        self.ipc_path = ipc_path
        self.fir_path = fir_path
        self.vectorizer = CountVectorizer(stop_words='english')
        self.merged_data = None
        self.count_matrix = None

    def load_and_prepare_data(self):
        ipc_data = pd.read_csv(self.ipc_path)
        fir_data = pd.read_csv(self.fir_path)

        ipc_data.fillna('', inplace=True)
        fir_data.fillna('', inplace=True)

        merged_data = ipc_data.copy()
        if {'Cognizable', 'Bailable', 'Court'}.issubset(fir_data.columns):
            merged_data = merged_data.merge(
                fir_data[['Description', 'Cognizable', 'Bailable', 'Court']],
                on='Description',
                how='left'
            )

        merged_data.fillna('', inplace=True)
        merged_data['Combined_Text'] = (
            merged_data['Description'] + ' ' +
            merged_data['Offense'] + ' ' +
            merged_data['Punishment'] + ' ' +
            merged_data.get('Cognizable', '') + ' ' +
            merged_data.get('Bailable', '') + ' ' +
            merged_data.get('Court', '')
        )

        self.merged_data = merged_data
        self.count_matrix = self.vectorizer.fit_transform(merged_data['Combined_Text'])

    def recommend(self, case_description, top_n=3, threshold=0.1):
        if self.count_matrix is None or self.merged_data is None:
            raise ValueError("Data not loaded. Call `load_and_prepare_data()` first.")

        case_vector = self.vectorizer.transform([case_description])
        similarities = cosine_similarity(case_vector, self.count_matrix)[0]

        recommendations = []
        for index, similarity in enumerate(similarities):
            if similarity > threshold:
                row = self.merged_data.iloc[index]
                recommendations.append((row['Section'], row['Description'], similarity))

        recommendations = sorted(recommendations, key=lambda x: x[2], reverse=True)[:top_n]
        return recommendations

    def save_model(self, model_path='model/ipc_model.joblib'):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump({
            'vectorizer': self.vectorizer,
            'count_matrix': self.count_matrix,
            'merged_data': self.merged_data
        }, model_path)

    def load_model(self, model_path='model/ipc_model.joblib'):
        model = joblib.load(model_path)
        self.vectorizer = model['vectorizer']
        self.count_matrix = model['count_matrix']
        self.merged_data = model['merged_data']
