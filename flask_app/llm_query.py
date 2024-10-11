import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the API key for Google Gemini
genai.configure(api_key=os.environ["API_KEY"])

# Define the function that queries the Gemini 1.5 model
def query_llm(prompt):
    try:
        # Use the Gemini 1.5 model to generate content
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Query the model using the input prompt
        response = model.generate_content(prompt)
        
        # Extract and return the generated text
        generated_text = response.text
        print(f"Generated Text: {generated_text}")
        
        return generated_text
    except Exception as e:
        print(f"Error querying LLM: {e}")
        return "An error occurred while processing your request."
