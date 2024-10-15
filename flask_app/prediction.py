from flask import Flask, request, render_template
import joblib  
import nltk
import json
import re
import string
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import logging
from llm_query import query_llm

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

# Setup logging
def log_error(e):
    logging.error(f"Error: {e}")

# Preprocessing functions
def first_preprocess(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input must be a non-empty string.")
    text = text.lower().strip()
    text = re.compile('<.*?>').sub('', text)  # Remove HTML tags
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)  # Remove punctuation
    text = re.sub('\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'\[[0-9]*\]', ' ', text)  # Remove numbers in brackets
    text = re.sub(r'\d', ' ', text)  # Remove digits
    return text

def stopword(string):
    return ' '.join([word for word in string.split() if word not in nltk.corpus.stopwords.words('english')])

wl = nltk.stem.WordNetLemmatizer()

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:
        return nltk.corpus.wordnet.NOUN

def lemmatizer(string):
    word_pos_tags = nltk.pos_tag(nltk.word_tokenize(string))
    return " ".join([wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for tag in word_pos_tags])

def data_preprocess(string):  # Fix function name
    return lemmatizer(stopword(first_preprocess(string)))

# Map text to emotions
def map_text_to_emotion(text, emotion_dict):
    words = text.split()
    matched_emotions = set()
    for word in words:
        if word in emotion_dict:
            matched_emotions.add(emotion_dict[word])
    return matched_emotions

def load_model():
    global best_model, emotion_dict, categories, sentence_model
    
    try:
        # Load the random forest model and emotion categories
        best_model = joblib.load('./model/rf_best.joblib')
        with open('./resources/categories_best.json', 'r') as json_file:
            categories = json.load(json_file)
        
        print(f"Categories loaded, type: {type(categories)}")
        
        emotion_df = pd.read_csv('./resources/Emotion Words.csv', header=None, names=['Word', 'Emotion'])
        emotion_replacement = {'surprise': 'joy', 'disgust': 'anger'}
        emotion_df['Emotion'] = emotion_df['Emotion'].replace(emotion_replacement)
        
        # Generate emotion dictionary
        emotion_word_dict = {row['Word']: row['Emotion'] for _, row in emotion_df.iterrows()}

    except Exception as e:
        log_error(f"An error occurred while loading resources: {e}")
        print(f"An error occurred while loading resources: {e}")
        return None, None

    try:
        # Load Sentence Transformer model
        sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        print("Sentence Transformer model loaded successfully.")
    except Exception as e:
        log_error(f"Failed to load Sentence Transformer model: {e}")
        print(f"Failed to load Sentence Transformer model: {e}")

    return best_model, categories, emotion_word_dict, sentence_model


# Load Sentence Transformer model for similarity

# Generate sentences based on emotions
# Generate sentences based on emotions using Gemini 1.5 API
def generate_sentences(input_text, emotions, num_sentences=5):
    print(emotions)
    sentences_dict = {}
    for emotion in emotions:
        prompt = f"Write {num_sentences} different sentences that express the emotion of {emotion}, based on this input: '{input_text}'."
        
        # Use the Gemini 1.5 API to generate content
        generated_text = query_llm(prompt)
        
        # Add the generated text to the sentences_dict for that emotion
        sentences_dict[emotion] = generated_text
    return sentences_dict

# Get most similar emotion based on cosine similarity
def get_most_similar_emotion(sentences_dict, new_comment_cleaned, sentence_model):
    print(sentences_dict, '\n',new_comment_cleaned)
    new_comment_embedding = sentence_model.encode(new_comment_cleaned, convert_to_tensor=True).cpu()
    max_similarity = -1
    most_similar_emotion = None

    for emotion, combined_sentences in sentences_dict.items():
        emotion_embedding = sentence_model.encode(combined_sentences, convert_to_tensor=True).cpu()
        similarity = cosine_similarity(new_comment_embedding.unsqueeze(0), emotion_embedding.unsqueeze(0))[0][0]
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_emotion = emotion
    return most_similar_emotion

# Prediction function
def emotion_prediction(text):
    best_model, categories, emotion_word_dict, sentence_model = load_model()
    processed_text = data_preprocess(text)
    print(processed_text)

    list_emotions = []  # Reset emotions list
    dict_emotions = map_text_to_emotion(text, emotion_word_dict)
    if dict_emotions:
        list_emotions = list(dict_emotions)
    
    vectorizer = best_model.named_steps['cv']
    sample_text_vectorized = vectorizer.transform([processed_text])  # Transform input

    # Make prediction
    predicted_label = best_model.named_steps['clf'].predict(sample_text_vectorized)
    predicted_emotion_label_cleaned = categories[predicted_label[0]]
    list_emotions.append(predicted_emotion_label_cleaned)

    # Generate sentences using the Gemini 1.5 API
    print(list_emotions, "hii")
    sentences_dict = generate_sentences(processed_text, list_emotions)
    
    # Find the most similar emotion based on cosine similarity
    most_similar_emotion = get_most_similar_emotion(sentences_dict, text, sentence_model)
    return most_similar_emotion

# Call the function

#text = 'I am happy to see you yesterday'
#most_similar_emotion = emotion_prediction(text)
#print(most_similar_emotion)
