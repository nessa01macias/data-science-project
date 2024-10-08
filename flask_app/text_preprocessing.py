import nltk

# Download stopwords if not already present
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')


def preprocess(text):
    # Convert text to lowercase
    text = text.lower()
    # Tokenize text into words
    words = nltk.word_tokenize(text)
    # Remove stop words
    stop_words = nltk.corpus.stopwords.words('english')
    words = [w for w in words if w not in stop_words]
    # Stem words using Porter stemmer
    stemmer = nltk.stem.PorterStemmer()
    words = [stemmer.stem(w) for w in words]
    # Return the processed text as a string
    return ' '.join(words)
