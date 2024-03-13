import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import nltk
nltk.download('punkt')

import nltk
nltk.download('popular')


#Create Training Data
training_data = {
    "greetings": {
        "inputs": ["hello", "hi"],
        "responses": ["hello", "Hi there!", "Greetings"]
    },  
    "goodbye" : {
        "inputs": ["bye", "goodbye"],
        "responses": ["Goodbye", "See you later","Jaa ne!","Bye","Sayonara"]
    }
}
#Process User Input
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()


def process_input(input_text):
    tokens = word_tokenize(input_text.lower())
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens

#Match User Input to Intents 
def  match_intent(processed_input):
    for intent, intent_data in training_data.items():
        for input_pattern in intent_data["inputs"]:
            if any(token in word_tokenize(input_pattern) for token in processed_input):
                return intent
    return None

#Generate Response 
import random 

def generate_response(intent):
    if intent in training_data:
        responses = training_data[intent]["responses"]
        return random.choice(responses)
    else:
        return "I'm sorry, I don't understand."

# Putting it All Together
def chatbot(input_text):
    processed_input = process_input(input_text)
    intent = match_intent(processed_input)
    response = generate_response(intent)
    return response

#Uncomment the line below to test the chatbot 
print(chatbot("bye"))