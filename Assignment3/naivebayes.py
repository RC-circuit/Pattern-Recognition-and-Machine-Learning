import numpy as np

def Naive_Bayes(data_df):
    
    spam_df = data_df[data_df['spam'] == 1]
    ham_df = data_df[data_df['spam'] == 0]

    spam_df.reset_index(drop=True, inplace=True)
    ham_df.reset_index(drop=True, inplace=True)

    p_hat = spam_df.shape[0]/data_df.shape[0]

    p_spam = []
    for i in range(len(spam_df['bag_of_words'][0])):
        sum = 0
        for tokens in spam_df['bag_of_words']:
            sum += tokens[i]
        
        p_spam.append(sum/spam_df.shape[0])
    

    p_ham = []
    
    for i in range(len(ham_df['bag_of_words'][0])):
        sum = 0
        for tokens in ham_df['bag_of_words']:
            sum += tokens[i]
        
        p_ham.append(sum/ham_df.shape[0])

    train_pred = []
    for text in data_df['bag_of_words']:
        ham_predict = 1-p_hat
        spam_predict = p_hat
        for j in range(len(ham_df['bag_of_words'][0])):
            if text[j] != 0:
                ham_predict *= p_ham[j]
                spam_predict *= p_spam[j]
            else:
                ham_predict *= (1-p_ham[j])
                spam_predict *= (1-p_spam[j])
        
        train_pred.append(spam_predict > ham_predict)

    accuracy = (train_pred == data_df['spam']).mean()
    print("Train Accuracy:", accuracy)

    return p_hat, p_ham, p_spam



def NaiveBayes_test(test_df, p_ham, p_spam, p_hat):
    test_pred = []
    for text in test_df['bag_of_words']:
        ham_predict = 1-p_hat
        spam_predict = p_hat
        for j in range(len(test_df['bag_of_words'][0])):
            if text[j] != 0:
                ham_predict *= p_ham[j]
                spam_predict *= p_spam[j]
            else:
                ham_predict *= (1-p_ham[j])
                spam_predict *= (1-p_spam[j])
        
        test_pred.append(spam_predict > ham_predict)

    accuracy = (test_pred == test_df['spam']).mean()
    print("Test Accuracy:", accuracy)

    return
