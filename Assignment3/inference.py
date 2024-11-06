import pickle
import numpy as np
import joblib
from dataset_prepare import data_preprocess, Bag_of_Words

import os

with open('vocabulary.pkl', 'rb') as f:
    vocabulary = pickle.load(f)

with open('p_hat.pkl', 'rb') as f:
    p_hat = pickle.load(f)

with open('p_ham.pkl', 'rb') as f:
    p_ham = pickle.load(f)

with open('p_spam.pkl', 'rb') as f:
    p_spam = pickle.load(f)


test_dir = './test'
files = [f for f in os.listdir(test_dir) if f.endswith('.txt')]





predictions = {}
for file in files:
    with open(os.path.join(test_dir, file), 'r') as f:
        text = f.read()
    processed_text = data_preprocess(text)
    features = np.zeros(len(vocabulary))
    for word in processed_text:
        if word in vocabulary:
            index = list(vocabulary).index(word)
            features[index] = 1

    ham_predict = 1-p_hat
    spam_predict = p_hat
    for j in range(len(vocabulary)):
        if features[j] != 0:
            ham_predict *= p_ham[j]
            spam_predict *= p_spam[j]
        else:
            ham_predict *= (1-p_ham[j])
            spam_predict *= (1-p_spam[j])
    
    label1 = int(spam_predict > ham_predict)
    predictions[file] = {'Naive Bayes': label1}
    
    
    svm_model = joblib.load('/home/ruthwik/Sem 6/PRML/assignment3/svm_pretrained.pkl')
    label2 = svm_model.predict(features.reshape(1,-1)).item()
    predictions[file]['Support Vector Machines'] = label2




for file, preds in predictions.items():
    print(f'{file}: Naive Bayes - {preds["Naive Bayes"]}, SVM - {preds["Support Vector Machines"]}')

