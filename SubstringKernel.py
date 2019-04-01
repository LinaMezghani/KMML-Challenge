import pandas as pd
import numpy as np

class SubstringKernel():
#     Input :
#          datafile : path to the data
#          list_k : list of the values of k for wich we wish to get the gram matrix for the k-LA-Kernel
#          lam : parameter of the LA-Kernel
#     Output of self.run() :
#          GRAM : gram matrices for all wanted values of k
    
    def __init__(self, datafile, list_k=[3], lam=0.6):
        self.max_k = max(list_k)
        self.list_k= list_k
        self.lam = lam
        self.data = pd.read_csv(datafile)
        self.N=len(self.data['seq'][0])
        self.init_matrices()
        self.GRAM={}
        for k in self.list_k:
            self.GRAM[k]=np.zeros((len(self.data['seq']), len(self.data['seq'])))
    
    def init_matrices(self):
        self.K = (self.max_k+1)*[(self.N+1)*[(self.N+1)*[-1]]]
        self.B = (self.max_k+1)*[(self.N+1)*[(self.N+1)*[-1]]]
    
    def main_K(self, x1, x2):
        self.init_matrices()
        for k in self.list_k:
            self.recurr_K(k, x1, x2)
        output={}
        for k in self.list_k:
            output[k]=self.K[k][-1][-1]
        return output
    
    def recurr_K(self, k, x1, x2):
        if self.K[k][len(x1)][len(x2)]==-1:
            if k==0:
                return 1
            elif min([len(x1), len(x2)])<k:
                return 0
            else:
                SUM=0
                for j in range(2, len(x2)):
                    if x2[j]==x1[-1]:
                        SUM+=self.recurr_B(k, x1, x2[:j-1])
                print(SUM)
                self.K[k][len(x1)][len(x2)]=(self.recurr_K(k, x1[:-1], x2)+SUM) 
        return self.K[k][len(x1)][len(x2)]
        
    def recurr_B(self, k, x1, x2):
        if self.B[k][len(x1)][len(x2)]==-1:
            if k==0:

                return 1
            elif min([len(x1), len(x2)])<k:
                return 0
            else:
                self.B[k][len(x1)][len(x2)]=(self.lam* self.recurr_B(k, x1[:-1], x2)+self.lam*self.recurr_B(k, x1, x2[:-1])
                                             - (self.lam**2) * self.recurr_B(k, x1[:-1], x2[:-1]) 
                                             + (x1[-1]==x2[-1])*(self.lam**2) * self.recurr_B(k-1, x1[:-1], x2[:-1])) 
        return self.B[k][len(x1)][len(x2)]
    
    def run(self):
        for i,x1 in enumerate(self.data['seq']):
            print("Calculating line "+ str(i))
            for j, x2 in enumerate(self.data['seq'][0:10]):
                if j%5==0:
                    print("... Column " + str(j))
                self.main_K(x1, x2)
                for k in self.list_k:
                    self.GRAM[k][i][j]=self.K[k][-1][-1]
        return self.GRAM