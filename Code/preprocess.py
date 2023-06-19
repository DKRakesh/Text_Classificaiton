# -*- coding: utf-8 -*-
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import torch

nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()



def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-z]', ' ', text)
    # Tokenize the text
    words = word_tokenize(text)
    # Remove stop words and stem the remaining words
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    # Join the words back into a single string
    text = ' '.join(words)
    return text

# Preprocess the text data


def main(dataframe):
    dataframe['text'] = dataframe['text'].apply(preprocess_text)
    

    labels = torch.tensor(dataframe['category'], dtype=torch.long)
    return dataframe['text'],labels
    
