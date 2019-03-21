import random

def convert_prediction(pred):
    if pred == -1 :
        return 0
    elif pred == 1 :
        return 1
    return random.choice([0,1])

def compute_val_accuracy(classifier, X_val, labels_val):
    acc = 0
    for i in range(X_val.shape[0]):
        if convert_prediction(classifier.predict(X_val[i])) == labels_val[i]:
            acc += 1
    return acc/X_val.shape[0]
