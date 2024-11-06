from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib
import ast
from dataset_prepare import Bag_of_Words, data_preprocess

def SVM_Classifier(train_df, test_df):

    X_train = list(train_df['bag_of_words'])
    y_train = train_df['spam']


    svm_classifier = SVC(kernel='linear')

    svm_classifier.fit(X_train, y_train)

    joblib.dump(svm_classifier, 'svm_pretrained.pkl')
    
    X_test = list(test_df['bag_of_words'])
    y_test = test_df['spam']
    
    y_pred = svm_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Test Accuracy:", accuracy)


    return svm_classifier.coef_, svm_classifier.intercept_


