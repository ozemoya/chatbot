import numpy as np

data = [
    ("How are you?", "greeting"),
    ("What is the weather today?", "weather"),
    ("Tell me a joke", "joke"),
    ("Goodbye", "farewell"),
    ("Hello", "greeting"),
    ("How's it going?", "greeting"),
    ("What's the weather like in New York today?", "weather"),
    ("Can you tell me a funny joke?", "joke"),
    ("Farewell, my friend", "farewell"),
    ("Hey, how are you doing?", "greeting"),
    ("Is it going to rain tomorrow?", "weather"),
    ("I need a good laugh", "joke"),
    ("Bye for now", "farewell"),
    ("Good day to you", "greeting"),
    ("Will it be sunny this weekend?", "weather"),
    ("Know any good jokes?", "joke"),
    ("See you later", "farewell"),
    ("Hi, nice to meet you", "greeting"),
    ("What's the temperature outside?", "weather"),
    ("Tell me something hilarious", "joke"),
    ("It was great seeing you", "farewell"),
    ("Hello, how have you been?", "greeting"),
    ("Do I need an umbrella today?", "weather"),
    ("Have you heard this one before?", "joke"),
    ("I'm off, goodbye!", "farewell"),
    ("Greetings, fellow human", "greeting"),
    ("Forecast for tomorrow?", "weather"),
    ("Give me your best joke", "joke"),
    ("Take care, goodbye!", "farewell")
    
]

#Feature Extraction 
from sklearn.feature_extraction.text import TfidfVectorizer
'''
TfidfVectorizer is a text vectorizer that converts a collection
of raw documents to a matrix of TF-IDF features.'''

#Extracting Features
texts = [text for text, intent in data]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

#Create Labels for Training
from sklearn.preprocessing import LabelEncoder 
'''
LabelEncoder is used to encode target labels with value between 0 and n_classes-1'''

#Encoding labels 
intents = [intent for text, intent in data]
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(intents)

#Train a Model 
from sklearn.linear_model import LogisticRegression
'''
LogisticRegression is a linear model for classification with
L2 regularization'''
model = LogisticRegression()
model.fit(X,y)

#Make Predictions
def predict_intent(text):
    text_features = vectorizer.transform([text])
    predicted_intent_index = model.predict(text_features)[0]
    predicted_intent = label_encoder.inverse_transform([predicted_intent_index])
    return predicted_intent[0]
print(predict_intent("How are you?"))

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Re-training the model with training set
model = LogisticRegression().fit(X_train, y_train)

# Predicting on the test set
y_pred = model.predict(X_test)

# Printing out the report
print(classification_report(
    y_test, 
    y_pred, 
    labels=np.unique(y_pred), 
    target_names=label_encoder.inverse_transform(np.unique(y_pred))
))

