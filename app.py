import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# Load data
df = pd.read_csv("emotion_dataset.csv")
df['cleaned_text'] = df['text'].apply(clean_text)

# Train model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['label']

model = LogisticRegression()
model.fit(X, y)

# UI
st.title("😊 Emotion Detection App")

st.write("Enter a sentence and detect the emotion")

user_input = st.text_input("Type here:")

if st.button("Predict Emotion"):
    if user_input == "":
        st.warning("Please enter some text")
    else:
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)
        
        st.success(f"Predicted Emotion: {prediction[0]}")