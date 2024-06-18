import re
import time
import numpy as np
import pandas as pd
import streamlit as st

# from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

start = time.time()

# Load the dataset
data = pd.read_csv('train.csv')
data.fillna(' ', inplace=True)
data['content'] = data['author'] + ' ' + data['title']
news_df = data[['content', 'label']]


# Text Preprocessing
ps = PorterStemmer()
def text_preprocessing(text):
    
    cleaned_data = re.sub('[^a-z A-Z]', ' ', text)
    lower_text = cleaned_data.lower()
    tokens = lower_text.split()
    stemmed_data = [ps.stem(word) for word in tokens]
    pure_str = ' '.join(stemmed_data)
    
    return pure_str

# Calling and applying this function to entire data
news_df['content'] = news_df['content'].apply(text_preprocessing)


# Vectorized Form
X = news_df['content'].values
y = news_df['label'].values
vector = TfidfVectorizer(stop_words='english')
vector.fit(X)
X = vector.transform(X)

# Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Building
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluation
print(accuracy_score(y_train, y_train_pred))
print(accuracy_score(y_test, y_test_pred))

print(f'{time.time() - start} seconds')


# Web-App using Streamlit
st.title('Fake News Detector')
input_text = st.text_input('Enter your News Article: ')

def prediction(text):

    input_data = vector.transform([text])
    pred = model.predict(input_data)

    return pred[0]

if input_text:
    final_pred = prediction(input_text)
   
    if final_pred == 0:
        st.write('Real News')
    else:
        st.write('Fake News')