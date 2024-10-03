from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import random

# Create SQLite database
engine = create_engine('sqlite:///quotes.db', echo=True)
Base = declarative_base()

# Define the Quote model
class Quote(Base):
    __tablename__ = 'quotes'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    quote = Column(String, nullable=False)
    author = Column(String, nullable=False)
    category = Column(String, nullable=False)
    score = Column(Integer, default=0)

# Create the table in the database
Base.metadata.create_all(engine)

# Create a session to interact with the database
Session = sessionmaker(bind=engine)
session = Session()

# Function to add a new quote
def add_quote(quote_text, author, category):
    new_quote = Quote(quote=quote_text, author=author, category=category)
    session.add(new_quote)
    session.commit()

# Function to get quotes by emotion/category
def get_quotes_by_category(category):
    return session.query(Quote).filter_by(category=category).all()

# Function to update quote score (for likes)
def update_quote_score(quote_id, increment=1):
    quote = session.query(Quote).filter_by(id=quote_id).first()
    if quote:
        quote.score += increment
        session.commit()
    return quote

# Recommendation system: Fetch a quote by category using weighted probabilities
def suggest_quote_by_category(category):
    quotes = get_quotes_by_category(category)
    if not quotes:
        return None
    
    # Use the score to create weighted probabilities
    scores = [quote.score + 1 for quote in quotes]  # Add 1 to avoid zero probabilities
    selected_quote = random.choices(quotes, weights=scores, k=1)[0]
    
    return {
        'id': selected_quote.id,
        'quote': selected_quote.quote,
        'author': selected_quote.author,
        'category': selected_quote.category,
        'score': selected_quote.score
    }
