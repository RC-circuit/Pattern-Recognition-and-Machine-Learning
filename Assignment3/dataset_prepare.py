import numpy as np
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')



def data_preprocess(text):

    rm_punct = ''.join([char for char in text if char not in string.punctuation])
    words = (rm_punct.lower()).split()
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english') and word.isalpha()]

    return tokens


def Bag_of_Words(data_df):

    vocabulary = list(set([word for sublist in data_df['preprocessed_text'] for word in sublist]))
    bag_of_words_embeddings = []
    for text in data_df['preprocessed_text']:
        bag_of_words = np.zeros(len(vocabulary))
        for word in text:
            if word in vocabulary:
                index = list(vocabulary).index(word)
                bag_of_words[index] = 1
        bag_of_words_embeddings.append(bag_of_words)

    return bag_of_words_embeddings, vocabulary


