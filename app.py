
import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Page Config
st.set_page_config(page_title="Spam Classifier | Ayush Developer")

# 1. Load the saved model and vectorizer
model = pickle.load(open('spam_model.pkl', 'rb'))
cv = pickle.load(open('vectorizer.pkl', 'rb'))

# 2. UI Setup
st.title("Email Spam Classifier")
user_input = st.text_area("Enter the message you want to check:")

if st.button("Predict"):
    if user_input:
        # 3. Transform input using the LOADED vectorizer
        data = [user_input]
        vectorized_data = cv.transform(data)
        
        # 4. Predict
        prediction = model.predict(vectorized_data)
        
        if prediction[0] == 1:
            st.success("This is HAM (Safe)")
        else:
            st.error("This is SPAM!")
    else:
        st.warning("Please enter a message.")
