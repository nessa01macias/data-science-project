import joblib
try:
    vectorizer = joblib.load('C:\\Users\\nessa\\Documents\\data-science-project\\flask_app\\vectorizer_reviews.joblib')
    print("Vectorizer loaded successfully.")
except Exception as e:
    print(f"Failed to load the vectorizer: {e}")
