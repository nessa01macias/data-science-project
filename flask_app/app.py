from flask import Flask, request, render_template, jsonify, url_for
from db_interaction import suggest_quote_by_category, update_quote_score  # Import from the database script
from llm_query import query_llm
from prediction import emotion_prediction, data_preprocess
from flask import redirect, url_for
import matplotlib.pyplot as plt
import io
import base64

# Initialize Flask app
app = Flask(__name__)

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

# New emotion mapping connecting original to new emotions
new_emotion_mapping = {
    "joy": "happy",
    "sadness": "sad",
    "shame": "sad",
    "guilt": "sad",
    "anger": "angry",
    "fear": "fearful",
    "disgusted": "disgusted",  # Same as original
    "neutral": "neutral",  # Same as original
}

last_predicted_emotion = None
current_quote = None

# Homepage route
@app.route("/")
def home():
    return redirect(url_for('page_of_quote'))

# Page of Quote
@app.route("/pageOfQuote", methods=["GET", "POST"])
def page_of_quote():
    global last_predicted_emotion, current_quote

    # Mapping of emotions to corresponding image URLs
    image_mapping = {
        "angry": url_for('static', filename="images/angry.png"),
        "disgusted": url_for('static', filename= "images/disgusted.jpg"),
        "fearful":url_for('static', filename= "images/fearful.png"),
        "happy": url_for('static', filename="images/happy.jpg"),
        "neutral": url_for('static', filename="images/neutral.png"),
        "sad": url_for('static', filename="images/sad.jpg")
    }

    if request.method == "POST":
        input_text = request.form.get('review', '')
        adjustment = request.form.get('adjustment', 'none')  # Fetch adjustment choice
        print(f"Review: {input_text}, Adjustment: {adjustment}")

        # Redirect to /pageOfRewrite if an adjustment is selected
        if adjustment != 'none':
            return render_template(
                "redirect_to_rewrite.html",
                review=input_text,
                adjustment=adjustment
            )

        try:
            # Preprocess the input data
            processed_input = data_preprocess(input_text)

            # Predict the emotion
            _predicted_emotion = emotion_prediction(input_text)
            predicted_emotion = new_emotion_mapping.get(_predicted_emotion)

            # Fetch a quote and song based on the predicted emotion
            quote_data = suggest_quote_by_category(predicted_emotion)
            if quote_data:
                quote = quote_data['quote']
                author = quote_data['author']
                last_predicted_emotion_quote_id = quote_data['id']
                stat_mapping[predicted_emotion] += 1  # Increment the stat for the predicted emotion
            else:
                quote = "No quote available."
                author = ""
                last_predicted_emotion_quote_id = None

            # Get the song corresponding to the predicted emotion
            song = song_mapping.get(predicted_emotion, "")

            # Select the image URL based on the predicted emotion
            image_url = image_mapping.get(predicted_emotion, url_for('static', filename="images/1.png"))  # Fallback image

            # Render the template with the prediction results
            return render_template(
                "pageOfQuote.html",
                prediction_text=f"The predicted emotion is {predicted_emotion}",
                quote_text=quote,
                song_text=song,
                last_predicted_emotion_quote_id=last_predicted_emotion_quote_id,
                review=input_text,
                predicted_emotion=predicted_emotion,
                image_url=image_url  # Pass the selected image URL to the template
            )

        except Exception as e:
            # Handle errors during processing and provide feedback
            return render_template(
                "pageOfQuote.html",
                prediction_text=f"Error in prediction: {str(e)}",
                review=input_text,
                quote_text="No quote available.",
                song_text="",
                image_url="https://example.com/default-image.jpg"  # Default image in case of error
            )

    # Render the page for GET requests
    return render_template("pageOfQuote.html")


# Page of Rewrite
@app.route("/pageOfRewrite", methods=["GET", "POST"])
def page_of_rewrite():
    if request.method == "POST":
        input_text = request.form.get('review', '')
        adjustment = request.form.get('adjustment', 'none')  # Fetch adjustment choice
        print(f"Review: {input_text}, Adjustment: {adjustment}")

        try:
            processed_input = data_preprocess(input_text)

            # Generate the suggestion (adjusted text)
            suggestion_text = None
            generated_example = None

            if adjustment != 'none' and len(input_text) > 5:  # Ensure enough input text to adjust
                # Generate the suggested version based on the adjustment type
                llm_prompt_suggestion = f"""
                Return your response in HTML format.

                Please make the following text {adjustment}. Keep the response concise and focused on specific, actionable feedback. Avoid offering revision examples or rewriting the text. 
                Provide 2-3 short, actionable suggestions for improvement, focusing on word choice and tone adjustments.
                
                Original text: {processed_input}
                    """
                suggestion_text = query_llm(llm_prompt_suggestion)
                suggestion_text = suggestion_text.replace("```html", "").replace("```", "").strip()

                print(f"Suggestion: {suggestion_text}")

                # Generate a new example using the adjustment as a basis
                llm_prompt_example = f"Make the following text {adjustment}. Create a new text based on this one but with the target emotion or style: {input_text}"
                generated_example = query_llm(llm_prompt_example)

            return render_template(
                "pageOfRewrite.html",
                original_text=input_text if adjustment != 'none' else None,
                modified_text=suggestion_text,
                generated_example=generated_example,
                adjustment=adjustment,
                review=input_text  # Keep the review text in the form for reuse
            )

        except Exception as e:
            return render_template(
                "pageOfRewrite.html",
                error_message=f"Error in processing: {str(e)}",
                review=input_text  # Keep the input text even on error
            )

    # Render the page for GET requests (without suggestions, just the form)
    return render_template("pageOfRewrite.html")


@app.route('/new-quote', methods=['POST'])
def new_quote():
    global current_quote
    data = request.get_json()
    action = data.get('action')
    quote_id = data.get('quote_id')
    emotion = data.get('emotion')

    if quote_id:
        if action == 'like':
            update_quote_score(quote_id, increment=1)
        elif action == 'dislike':
            update_quote_score(quote_id, increment=-1)

    if emotion:
        quote_data = suggest_quote_by_category(emotion)
        if quote_data and quote_data['quote'] != current_quote:
            current_quote = quote_data['quote']
            return jsonify({'new_quote': quote_data['quote'], 'author': quote_data['author']})
        else:
            return jsonify({'new_quote': "No new quote available."})

    return jsonify({'new_quote': "No emotion specified."})


if __name__ == "__main__":
    app.run(debug=True)