from db_interaction import suggest_quote_by_category, update_quote_score  # Import from the database script

from flask import Flask, request, render_template, jsonify
import pickle  # For loading the model
import joblib  # For loading the vectorizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
import io
import base64
from dotenv import load_dotenv
import os 
import random

load_dotenv()

vectorizer_path = os.getenv('VECTORIZER_PATH')

# Download stopwords if not already present
nltk.download('stopwords')

# Initialize Flask app
app = Flask(__name__)

# Load the Naive Bayes model from the pickle file
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
    vectorizer = joblib.load(vectorizer_path)
except FileNotFoundError:
    print(f"Vectorizer file not found. Please check the path: {vectorizer_path}")
    vectorizer = None
except Exception as e:
    print(f"An error occurred while loading the vectorizer: {e}")
    vectorizer = None

# Define a mapping of predicted class labels to emotions
emotion_mapping = {
    0: "happy",
    1: "sad",
    2: "angry",
    3: "surprised",
    4: "fearful",
    5: "disgusted",  # Changed from disgusted to disgust
    6: "neutral"
}

# Define a mapping for playlists
song_mapping = {
    "happy": "https://open.spotify.com/playlist/7INcD4lmarWTQiDVodjVt4",
    "sad": "https://open.spotify.com/playlist/5iTgdtVp3FgHfFHFqkyss2",
    "angry": "https://open.spotify.com/playlist/0a4Hr64HWlxekayZ8wnWqx",
    "surprised": "https://open.spotify.com/playlist/6cuwjdJXzkS12E3GoAVHdu",
    "fearful": "https://open.spotify.com/playlist/405PXg1fJunIlSo75L78Kb",
    "disgusted": "https://open.spotify.com/playlist/1MPXG4at633jUvtw7lLiOG",
    "neutral": "https://open.spotify.com/playlist/6C4VTR3wHa8rQLV72RMG1f"
}

stat_mapping = {
    "happy": 0,
    "sad": 0,
    "angry": 0,
    "surprised": 0,
    "fearful": 0,
    "disgusted": 0,
    "neutral": 0
}

last_predicted_emotion = None
current_quote = None

# Homepage route
@app.route("/")
def home():
    return render_template("index1.html")

@app.route('/predict', methods=["POST"])
def predict():
    global last_predicted_emotion, current_quote
    review = request.form['Review']  # Get the review text from the form input

    # Step 1: Preprocess the review (same as before)
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
        print("The predicted emotion is:", predicted_emotion)
        # Fetch a quote from the database using the predicted emotion
        quote_data = suggest_quote_by_category(predicted_emotion)
        print("the quote data is", quote_data)

        if quote_data:
            quote = quote_data['quote']
            author = quote_data['author']
            last_predicted_emotion_quote_id = quote_data['id']  # Store the quote ID for feedback
            stat_mapping[predicted_emotion] += 1
        else:
            quote = "No quote available."
            author = ""
            last_predicted_emotion_quote_id = None

        song = song_mapping[predicted_emotion]
    except Exception as e:
        predicted_emotion = "Error in prediction: " + str(e)
        quote = "No quote available."
        author = ""
        song = ""
        last_predicted_emotion_quote_id = None

    current_quote = quote

    return render_template("index1.html", 
                        prediction_text=f"The predicted emotion is: {predicted_emotion}",
                        quote_text=quote, 
                        author_text=author, 
                        song_text=song, 
                        last_predicted_emotion_quote_id=last_predicted_emotion_quote_id,
                        predicted_emotion=predicted_emotion)  # Pass the predicted emotion

# Statistics route
@app.route('/statistics')
def stats():
    filtered_data = {emotion: count for emotion, count in stat_mapping.items() if count > 0}
    if not filtered_data:
        return render_template('stats.html', message="No data available")

    labels = list(filtered_data.keys())
    amounts = list(filtered_data.values())
    colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0','#ffb3e6','#ff6666']

    fig, ax = plt.subplots()
    ax.pie(amounts, labels=labels, colors=colors, shadow=True)
    ax.axis('equal')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    chart_url = base64.b64encode(img.getvalue()).decode('utf-8')

    return render_template('stats.html', chart_url=chart_url)

@app.route('/new-quote', methods=['POST'])
def new_quote():
    global current_quote
    data = request.get_json()
    action = data.get('action')
    quote_id = data.get('quote_id')  # Get the quote_id from the request
    emotion = data.get('emotion')  # Get the emotion from the frontend

    print(f"Action: {action}, Quote ID: {quote_id}, Emotion: {emotion}")

    # If the action is 'like' or 'dislike', update the score accordingly
    if quote_id:
        if action == 'like':
            update_quote_score(quote_id, increment=1)  # Increment score
        elif action == 'dislike':
            update_quote_score(quote_id, increment=-1)  # Decrement score

    # Fetch a new quote from the database for the emotion passed
    if emotion:  # Ensure emotion is not empty
        quote_data = suggest_quote_by_category(emotion)
        if quote_data and quote_data['quote'] != current_quote:
            current_quote = quote_data['quote']
            return jsonify({
                'new_quote': quote_data['quote'],
                'author': quote_data['author']
            })
        else:
            return jsonify({'new_quote': "No new quote found."})
    else:
        return jsonify({'error': 'No emotion provided.'})


# Route to handle likes and dislikes
@app.route('/update-score', methods=['POST'])
def update_score():
    global current_quote_id
    data = request.get_json()
    action = data.get('action')

    if current_quote_id is not None:
        # If the action is 'like', increment the score, otherwise decrement
        increment = 1 if action == 'like' else -1
        updated_quote = update_quote_score(current_quote_id, increment)

        return jsonify({
            'quote': updated_quote.quote,
            'author': updated_quote.author,
            'new_score': updated_quote.score
        })
    return jsonify({'error': 'No quote available to update score.'}), 400


# Run the app
if __name__ == "__main__":
    app.run(debug=True)
