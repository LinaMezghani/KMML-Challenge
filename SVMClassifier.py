import numpy as np

class SVMClassifier :
    
    def __init__(self, kernel, C) :
        self.kernel = kernel
        self.C = C
        print("C : {}".format(self.C))
        self.max_iter = 20
        print("max_iter : {}".format(self.max_iter))
        self.tol = 0.001
        print("tol : {}".format(self.tol))
        self.alphatol = 0.0001
        print("alphatol : {}".format(self.alphatol))
    
    def compute_gram_matrix(self, X):
        n=X.shape[0]
        K=np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                K[i,j] = self.kernel(X[i], X[j])
        return K
    
    def minimize_dual(self, X, Yoriginal) :
        Y = 2*Yoriginal -1
        #self.X = X
        self.Y = Y
        b=0
        n,d=X.shape[0], X.shape[1]
        alphas = np.zeros(n)
        iterations = 0
        
        while(iterations<self.max_iter) :
            
            #print("iteration number : {}".format(iterations))
            num_changed_alphas = 0
            for i in range(n):
                #time.sleep(0.1)
                #print("SEPPPPPPPPPPPPPPPPPPPPP")
                #print("i : {}".format(i))
                #print("Yi : {}".format(Y[i]))
                Ei =  b+ np.dot(np.multiply(Y,alphas), self.K[i,:]) - Y[i]
                #print("Ei : {}".format(Ei))
                if ((Y[i]*Ei<-self.tol) and (alphas[i]<self.C)) or ((Y[i]*Ei>self.tol) and (alphas[i]>0)) :
                    #print("got here")
                    j=self.select_randomly_j(i,n)
                    #print("j : {}".format(j))
                    Ej=  b+ np.dot(np.multiply(Y,alphas), self.K[j,:]) - Y[j]
                    #print("Yj : {}".format(Y[j]))
                    #print("Ej : {}".format(Ej))
                    kernel_ij = self.K[i,j]
                    #print("Kij : {}".format(kernel_ij))
                    alphaoldi = alphas[i]
                    alphaoldj = alphas[j]
                    if Y[i]!=Y[j] :
                        L =max(0, alphas[j] - alphas[i])
                        H = min(self.C, self.C + alphas[j] - alphas[i])
                    else :
                        L = max(0,alphas[i] + alphas[j] - self.C )
                        H =min(self.C, alphas[i] +alphas[j])
                    #print("L : {}".format(L))
                    #print("H : {}".format(H))
                    #print("alphaj : {}".format(alphas[j]))
                    #print("alphai : {}".format(alphas[i]))
                    if L==H :
                        continue
                    else :
                        eta = 2*kernel_ij - self.K[i,i] - self.K[j,j]
                        #print("eta : {}".format(eta))
                        if eta>=0:
                            continue
                        else :
                            #print("unclipped : {}".format( alphas[j] - (1/eta)* Y[j] * (Ei-Ej)))
                            newalphaj =np.clip( alphas[j] - ((1/eta)* Y[j] * (Ei-Ej)) , L,H)
                            #print(newalphaj)
                            #print("newalphaj : {}".format(newalphaj))
                            alphas[j] = newalphaj
                            if np.abs(alphas[j] - alphaoldj) < self.alphatol :
                                continue
                            else :
                                newalphai= alphas[i]+Y[i]*Y[j]*(alphaoldj - alphas[j])
                                alphas[i] = newalphai
                                b1 = b-Ei-Y[i]*(alphas[i]-alphaoldi)*self.K[i,i] - Y[j] *(alphas[j]-alphaoldj)*kernel_ij
                                b2 = b-Ej-Y[i]*(alphas[i]-alphaoldi)*kernel_ij - Y[j] *(alphas[j]-alphaoldj)*self.K[j,j]
                                if alphas[i]>0 and alphas[i] < self.C :
                                    b=b1
                                elif  alphas[j]>0 and alphas[j] < self.C :
                                    b=b2
                                else :
                                    b= (b1+b2)/2
                                num_changed_alphas +=1
            if num_changed_alphas ==0 :
                iterations += 1
            else :
                iterations = 0
        self.b= b
        return alphas
    
    def fit(self, X, Y) :
        print("computing gram matrix")
        self.K = self.compute_gram_matrix(X)
        print("gram matrix computed")
        self.X = X
        self.Y = 2*Y-1
        self.alphas= self.minimize_dual(X, Y)
        support_vector_indexes = (self.alphas) > self.tol
        support_vector2 = self.alphas < self.C+ self.tol
        support_vector_indexes = np.multiply(support_vector_indexes, support_vector2)
        new_alphas = np.zeros(X.shape[0])
        for i in range(X.shape[0]) :
            new_alphas[i]= 0
            if self.alphas[i] > self.tol  and self.alphas[i] <  self.C + self.tol:
                new_alphas[i]  = self.alphas[i]
        self.new_alphas = new_alphas
        print(support_vector_indexes)
        print("proportion of support vectors : {}".format( np.mean(support_vector_indexes)))
    
    def predict(self, test_point) :
        Kpoint = np.array([self.kernel(test_point, self.X[i]) for i in range(self.X.shape[0])])
        value  = np.dot(np.multiply(self.Y,self.new_alphas), Kpoint)
        return np.sign(value + self.b)
    
    def select_randomly_j(self, i,n):
        t=i
        while (t==i):
            t=np.random.randint(n)
        return t
