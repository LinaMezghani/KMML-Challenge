import random
from SpectrumKernel import SpectrumKernel
import csv

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

def to_vect_test(datafile, numbers_list, dictionnary=True):
    # Takes a dataframe, returns the corresponding dataframe containing :
    # The dictionaries of k-grams corresponding to each line if dictionnary = True
    # Vectors corresponding to the dictionnaries if dictionnary = False
    kernel = SpectrumKernel(datafile, numbers_list)
    data = kernel.special_vectorization()
    return data


def generate_predictions(test_file, path, numbers_list, classifier = False):
    test_data = to_vect_test(path + test_file,numbers_list)
    predictions = []
    for i in range(len(test_data)):
        emb_seq = test_data[i]
        prediction = (classifier.predict(emb_seq))
        prediction = convert_prediction(prediction)
        predictions.append(int(prediction))
    return predictions



def generate_submission_file(classifiers, path, numbers_list, submission_filename = 'submission.csv'):
    test_files = ['Xte0.csv', 'Xte1.csv', 'Xte2.csv']
    predictions = []
    for i in range(len(test_files)) :
        predictions += generate_predictions(test_files[i], path, numbers_list, classifiers[i])
    i = 0
    with open(submission_filename, mode='w') as submission_file:
        writer = csv.writer(submission_file, delimiter=',')
        writer.writerow(['Id', 'Bound'])
        for prediction in predictions :
            writer.writerow([str(i), prediction])
            i += 1
