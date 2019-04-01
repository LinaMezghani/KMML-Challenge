import pandas as pd
import numpy as np


gapped = False 
nucleotides =["A", "T","G", "C"]
nucleotide_gaps = dict()
for i in range(4):
    for j in range(4): 
        nucleotide_gaps[nucleotides[i]+ nucleotides[j]+"2"]=i*4+j
        nucleotide_gaps[nucleotides[i]+ nucleotides[j]+"3"]=16 +i*4+j


def gaps(string_test) :
    vector=np.zeros(32)
    for i in range(0,len(string_test)-2):
        test_element= string_test[i:i+3]
        vector[nucleotide_gaps[test_element[0]+test_element[2]+"2"]] +=1
    for i in range(0,len(string_test)-3):
        test_element= string_test[i:i+4]
        vector[nucleotide_gaps[test_element[0]+test_element[3]+"3"]] +=1
    return vector 


def gaps_vectors(data) : 
    vectors = np.zeros((data.shape[0], 32))
    for ind, seq in enumerate(data["seq"]) : 
        vectors[ind] = gaps(seq)
    return vectors

class VectorizedDataset :
    """Parameters :
        - rpz : the representation on which to build the dataset
        - datapath : the path of data
        - list_k : parameters for the rpz
        """
    def __init__(self, rpz, data_path, list_k):
        self.representation = rpz
        self.datapath = data_path
        self.list_k = list_k
        self.X, self.Y = self.build_XY()
    
    def build_X_labels(self, fname_X, fname_labels):
        X = self.representation(self.datapath + fname_X, self.list_k).vectorization
        if gapped : 
            data = pd.read_csv(self.datapath+fname_X)
            X2 = gaps_vectors(data)
            X= np.hstack([X,X2])

        labels = pd.read_csv(self.datapath + fname_labels)
        labels =np.array(labels["Bound"])
        return X, labels
    
    def build_train_val(self, X, labels, shuffle = True):
        if shuffle :
            data = np.c_[X.reshape(len(X),-1), labels.reshape(len(labels), -1)]
            np.random.shuffle(data)
            X = data[:, :X.size//len(X)].reshape(X.shape)
            labels = data[:, X.size//len(X):].reshape(labels.shape)
        X_train, X_val = np.split(X, [int(0.9*X.shape[0])])
        labels_train, labels_val = np.split(labels, [X_train.shape[0]])
        return X_train, X_val, labels_train, labels_val
    
    def build_XY(self):
        X = {}
        labels = {}
        for i in range(3) :
            X_tmp, labels_tmp = self.build_X_labels('Xtr{}.csv'.format(i), 'Ytr{}.csv'.format(i))
            X[i] = {}
            labels[i] = {}
            X[i]['train'], X[i]['val'], labels[i]['train'], labels[i]['val'] = self.build_train_val(X_tmp, labels_tmp)
        return X, labels
