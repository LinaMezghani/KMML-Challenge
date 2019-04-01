import itertools
import pandas as pd
import numpy as np


class SpectrumKernel() :
    """Parameters :
        - list_k : the list that represents the k-grams we want to build the representation on ex: [2,3,4]
        - datafile : the name of the file to build the representation on
        """
    def __init__(self, datafile, list_k):
        self.list_k = list_k
        self.datafile = datafile
        self.vectorization = self.special_vectorization(datafile, list_k)
    
    def gramizer(self, sequence, k=3): # Takes a sequence, returns the dictionary of its k-grams
        assert len(sequence)>=k, "Sequence is shorter than the window"
        vect={}
        for cursor in range(k, len(sequence)+1):
            gram = sequence[cursor-k:cursor]
            if gram in vect:
                vect[gram]+=1
            else:
                vect[gram]=1
        return vect

    def to_vect(self, datafile, k=3, dictionnary=True):
        # Takes a dataframe, returns the corresponding dataframe containing :
        # The dictionaries of k-grams corresponding to each line if dictionnary = True
        # Vectors corresponding to the dictionnaries if dictionnary = False
        data = pd.read_csv(datafile)
        data['seq']=data['seq'].apply(self.gramizer, args=(k,))
        if not dictionnary:
            ind_dict=self.gram_ind(k)
            data['seq']=data['seq'].apply(self.dict_to_vec, args=(ind_dict, k,))
            return data

    def gram_ind(self, k):
        permutations = list(itertools.product(['A', 'C', 'T', 'G'], repeat=k))
        string_to_number_dict = {}
        for ind, sequence in enumerate(permutations) :
            string_to_number_dict[''.join(sequence)] = ind
        return string_to_number_dict

    def dict_to_vec(self, dico, ind_dict, k):
        vect=np.zeros(len(ind_dict))
        for key, value in dico.items():
            vect[ind_dict[key]]=value
        return vect

    def vec_to_table(self, data):
        N= data.shape[0]
        M= data['seq'][0].size
        output = np.zeros((N,M))
        bias=data['Id'][0]
        index=0
        for i in data['Id']:
            output[index]=data['seq'][i-bias]
            index+=1
        return output

    def special_vectorization (self, datafile, list_k, in_dict=False) :
        tables = []
        for number in list_k:
            tables.append(self.vec_to_table(self.to_vect(datafile, number, in_dict)))
        return np.hstack(tables)
