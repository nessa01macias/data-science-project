import requests
import time
from db_interaction import add_quote
from dotenv import load_dotenv
import os 
import random

load_dotenv()

MY_OWN_API = os.getenv('MY_OWN_API')

# Define the emotion categories and their mapped API categories
categories = {
    "Neutral": ["life", "inspirational", "hope", "future", "imagination"],
    "Joy": ["happiness", "love", "friendship", "funny", "humor", "success", "great", "good"],
    "Sadness": ["alone", "failure", "death"],
    "Fear": ["fear", "courage"],
    "Surprise": ["amazing", "cool"],
    "Anger": ["anger", "jealousy"],
    "Shame": ["failure"],
    "Disgust": ["anger"],
}

# Function to fetch quotes from the API based on emotion category
def get_quote_by_emotion(emotion):
    if emotion not in categories:
        print(f"No categories mapped for emotion: {emotion}")
        return None
    
    # Shuffle categories for randomness
    emotion_categories = categories[emotion]
    random.shuffle(emotion_categories)
    
    for category in emotion_categories:
        api_url = f'https://api.api-ninjas.com/v1/quotes?category={category}'
        response = requests.get(api_url, headers={'X-Api-Key': MY_OWN_API})
        if response.status_code == requests.codes.ok:
            data = response.json()
            if data:
                quote_text = data[0]['quote']
                author = data[0]['author']
                return {
                    'quote': quote_text,
                    'author': author,
                    'category': emotion
                }
        else:
            print(f"Error: {response.status_code} {response.text}")
    print(f"No quotes found for emotion: {emotion}")
    return None

# Function to populate quotes into the database
def populate_quotes():
    emotions = categories.keys()
    for emotion in emotions:
        for _ in range(10):  # Fetch 10 quotes per emotion
            quote_data = get_quote_by_emotion(emotion)
            if quote_data:
                add_quote(
                    quote_text=quote_data['quote'],
                    author=quote_data['author'],
                    category=quote_data['category']
                )
                print(f"Added quote for emotion: {emotion}")
            time.sleep(5)  # Sleep for 2 seconds to avoid rate-limiting

# Run the populate function
if __name__ == '__main__':
    populate_quotes()
