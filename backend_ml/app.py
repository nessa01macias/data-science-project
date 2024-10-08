# app.py

from flask import Flask, request, render_template
import pickle  # For loading the model
import joblib  # For loading the vectorizer
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load the Naive Bayes model from pickle file
with open('/model/gnb_model.pkl', 'rb') as file:
    gnb_from_pickle = pickle.load(file)
    
print("Naive Bayes model loaded successfully.")

# Initialize Flask app
app = Flask(__name__)

# Load the vectorizer using joblib
vectorizer = joblib.load('vectorizer_reviews.joblib')

# Define a mapping of predicted class labels to emotions
emotion_mapping = {
    0: "happy",
    1: "sad",
    2: "angry",
    3: "surprised",
    4: "fearful",
    5: "disgusted",
    6: "neutral"
}

# Homepage route
@app.route("/")
def home():
    return render_template("index1.html")

# Prediction route
@app.route('/predict', methods=["POST"])
def predict():
    review = request.form['Review']  # Get the review text from the form input
    
    # Step 1: Preprocess the review (similar to training preprocessing)
    review = re.sub('[^a-zA-Z]', ' ', review).lower().split()
    ps = PorterStemmer()
    stopwords_list = stopwords.words('english')
    
    # Add custom stopwords that were kept in the training phase
    custom_stopwords = ['not', 'no', 'but', "won't", 'too', 'very']
    for word in custom_stopwords:
        stopwords_list.remove(word)
    
    # Stem the words and remove stopwords
    review = [ps.stem(word) for word in review if word not in set(stopwords_list)]
    review = ' '.join(review)
    
    # Step 2: Vectorize the preprocessed review
    review_vectorized = vectorizer.transform([review])
    
    # Step 3: Predict the emotion using the loaded Naive Bayes model
    prediction = gnb_from_pickle.predict(review_vectorized)
    
    # Step 4: Map the predicted class label to the corresponding emotion
    predicted_emotion = emotion_mapping[prediction[0]]
    
    # Render the result in the HTML template
    return render_template("index1.html", prediction_text="The predicted emotion is: {}".format(predicted_emotion))

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
