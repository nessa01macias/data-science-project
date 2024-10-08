from flask import Flask, request, render_template
import pickle  # For loading the model
import joblib  # For loading the vectorizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords if not already present
nltk.download('stopwords')

# Initialize Flask app
app = Flask(__name__)

# Load the Naive Bayes model from pickle file
try:
    with open('model/gnb_model.pkl', 'rb') as file:
        gnb_from_pickle = pickle.load(file)
    print("Naive Bayes model loaded successfully.")
except FileNotFoundError:
    print("Model file not found. Please check the path.")
except Exception as e:
    print(f"An error occurred while loading the model: {e}")

# Load the vectorizer using joblib
try:
    vectorizer = joblib.load('vectorizer_reviews.joblib')
except FileNotFoundError:
    print("Vectorizer file not found. Please check the path.")
except Exception as e:
    print(f"An error occurred while loading the vectorizer: {e}")

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
    return render_template("/index1.html")

# Prediction route
@app.route('/predict', methods=["POST"])

def predict():
    review = request.form['Review']  # Get the review text from the form input
    
    # Step 1: Preprocess the review (similar to training preprocessing)
    review = re.sub('[^a-zA-Z]', ' ', review).lower().split()
    ps = PorterStemmer()
    stopwords_list = set(stopwords.words('english'))
    
    # Add custom stopwords that were kept in the training phase
    custom_stopwords = ['not', 'no', 'but', "won't", 'too', 'very']
    stopwords_list.difference_update(custom_stopwords)
    
    # Stem the words and remove stopwords
    review = [ps.stem(word) for word in review if word not in stopwords_list]
    review = ' '.join(review)
    
    # Step 2: Vectorize the preprocessed review
    review_vectorized = vectorizer.transform([review]).toarray()  # Convert to dense array
    
    # Step 3: Predict the emotion using the loaded Naive Bayes model
    try:
        prediction = gnb_from_pickle.predict(review_vectorized)
        # Step 4: Map the predicted class label to the corresponding emotion
        predicted_emotion = emotion_mapping[prediction[0]]
    except Exception as e:
        predicted_emotion = "Error in prediction: " + str(e)
    
    # Render the result in the HTML template
    return render_template("index1.html", prediction_text="The predicted emotion is: {}".format(predicted_emotion))

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
