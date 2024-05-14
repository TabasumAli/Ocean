from django.shortcuts import render
from transformers import BertForSequenceClassification, BertTokenizer
import torch

# Define the directory to load the model
model_dir = "C:/Users/Hackathon/Downloads/Compressed/oceanTraitModelMay10"
loaded_model = BertForSequenceClassification.from_pretrained(model_dir)
tokenizer = BertTokenizer.from_pretrained(model_dir + '/config.json')

# Define the Ocean traits
ocean_traits = ['Extraversion (cEXT)', 'Neuroticism (cNEU)', 'Agreeableness (cAGR)', 'Conscientiousness (cCON)', 'Openness (cOPN)']

# Import necessary libraries for scraping, preprocessing, and model prediction
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import re
import tensorflow as tf
from django.http import HttpResponse

import requests

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to preprocess text
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens]
    tokens = [token for token in tokens if token not in string.punctuation]
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    processed_text = ' '.join(lemmatized_tokens)

    return processed_text


# Function to perform model prediction
def predict_personality_traits(input_data):
    # Your model prediction logic here
    return traits_dict

def evaluate_input(request):
    if request.method == 'POST':
        # Get the form data from the request
        input_data = {
            'input1': request.POST.get('input1', ''),
            'input2': request.POST.get('input2', ''),
            'input3': request.POST.get('input3', ''),
            'input4': request.POST.get('input4', ''),
            'input5': request.POST.get('input5', ''),
        }
        
        # Preprocess the input data
        preprocessed_data = preprocess_text(input_data)

        # Perform model prediction
        traits_dict = predict_personality_traits(preprocessed_data)

        # Render the result page with the evaluation result
        return render(request, 'oceantraits/traits.html', {'result': traits_dict})

def scrape_and_evaluate(request):
    if request.method == 'POST':
        # Get the profile URL from the request
        profile_url = request.POST.get('profile_url', '')

        # Scrape the profile data
        try:
            response = requests.get(profile_url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                # Assuming posts are within a certain class or tag, you need to find and extract them here
                # For example, if posts are within divs with class "_1xnd":
                posts_container = soup.find_all('div', class_='_1xnd')
                posts = [post.get_text(separator='\n', strip=True) for post in posts_container]

                # Preprocess the scraped data
                preprocessed_data = preprocess_text('\n'.join(posts))

                # Perform model prediction
                traits_dict = predict_personality_traits(preprocessed_data)

                # Render the result page with the evaluation result
                return render(request, 'oceantraits/traits.html', {'result': traits_dict})
            else:
                return HttpResponse("Failed to fetch profile data. Please check the URL.")
        except Exception as e:
            return HttpResponse(f"An error occurred: {str(e)}")




# oceantraits/views.py

def homepage(request):
    return render(request, 'oceantraits/homepage.html')

def evalutepg1(request):
    return render(request, 'oceantraits/evalutepg1.html')

def private_view(request):
    if request.method == 'POST':
        return scrape_and_evaluate(request)
    else:
        return render(request, 'oceantraits/private.html')

def public_view(request):
    if request.method == 'POST':
        return evaluate_input(request)
    else:
        return render(request, 'oceantraits/public.html')
