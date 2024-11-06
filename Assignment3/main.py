import numpy as np
import pandas as pd
import pickle
from dataset_prepare import data_preprocess, Bag_of_Words
from sklearn.model_selection import train_test_split
from naivebayes import Naive_Bayes, NaiveBayes_test
from SVM import SVM_Classifier

data_df = pd.read_csv('/home/ruthwik/Sem 6/PRML/assignment3/emails.csv')
data_df.drop_duplicates(inplace=True)
data_df["preprocessed_text"] = data_df["text"].apply(data_preprocess)
data_df['bag_of_words'], vocabulary = Bag_of_Words(data_df)


train_df, test_df = train_test_split(data_df, test_size=0.35, random_state=42)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

p_hat, p_ham, p_spam = Naive_Bayes(train_df)
NaiveBayes_test(test_df, p_ham, p_spam, p_hat)
weights, bias = SVM_Classifier(train_df, test_df)

with open('vocabulary.pkl', 'wb') as f:
    pickle.dump(vocabulary, f)

with open('p_hat.pkl', 'wb') as f:
    pickle.dump(p_hat, f)

with open('p_ham.pkl', 'wb') as f:
    pickle.dump(p_ham, f)

with open('p_spam.pkl', 'wb') as f:
    pickle.dump(p_spam, f)
