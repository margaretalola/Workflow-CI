import pandas as pd
import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import mlflow
import mlflow.sklearn
import joblib
import os
import logging
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

mlflow.set_experiment("Course-Recommendation")
mlflow.autolog() 

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df = df.dropna(subset=['Course Title'])
        if 'Review' in df.columns:
            df['Review'] = df['Review'].fillna(0)
        if 'What you will learn' in df.columns:
            df['What you will learn'] = df['What you will learn'].fillna('')
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None

def create_content_column(df):
    try:
        df['content'] = df['Course Title'] + ' ' + df['What you will learn']
        return df
    except Exception as e:
        logging.error(f"Error creating content column: {e}")
        return df

class DummyModel(BaseEstimator):
    def __init__(self, tfidf_vectorizer, cosine_matrix):
        self.tfidf_vectorizer = tfidf_vectorizer
        self.cosine_matrix = cosine_matrix

    def predict(self, X):
        return ["Not Implemented"] * len(X)
    

def train_model(df):
    with mlflow.start_run(nested=True):
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))
        tfidf_matrix = tfidf.fit_transform(df['content'])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Simpan model & cosine similarity matrix
        os.makedirs("models", exist_ok=True)
        joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
        joblib.dump(csr_matrix(cosine_sim), "models/cosine_sim_matrix.pkl")

        mlflow.log_artifact("models/tfidf_vectorizer.pkl")
        mlflow.log_artifact("models/cosine_sim_matrix.pkl")

         # Logging model ke MLflow
        dummy_model = DummyModel(tfidf, cosine_sim)
        mlflow.sklearn.log_model(dummy_model, artifact_path="model")

        logging.info("Model dan matriks similarity disimpan dan dicatat oleh MLflow")

def recommend_courses(course_title, df, cosine_sim, top_n=5):
    try:
        distances = [Levenshtein.distance(course_title.lower(), title.lower()) for title in df['Course Title']]
        closest_index = distances.index(min(distances))
        best_match_title = df['Course Title'].iloc[closest_index]
        course_index = df[df['Course Title'] == best_match_title].index[0]

        similarity_scores = list(enumerate(cosine_sim[course_index]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        top_courses = similarity_scores[1:top_n+1]
        course_indices = [i[0] for i in top_courses]
        return df['Course Title'].iloc[course_indices].tolist()
    except IndexError:
        logging.warning(f"Course containing '{course_title}' not found in the dataset.")
        return []

if __name__ == "__main__":
    data_path = "preprocessed_data.csv"
    df = load_data(data_path)
    if df is not None:
        df = create_content_column(df)
        train_model(df)