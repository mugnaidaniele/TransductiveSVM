import numpy as np
import sklearn.svm as svm
import time
class TransductiveSVM(svm.SVC):
    def __init__(self,kernel="rbf",Cl=1,Cu=0.01,gamma=0.1,X2=None):
        '''
        Initial TSVM
        Parameters
        ----------
        kernel: kernel of svm
        Cl: Penalty Inductive SVM
        Cu: Penalty Unlabeled set
        gamma: gamma for rbf kernel
        X2: Unlabeled set(only features)
                np.array, shape:[n2, m], n2: numbers of samples without labels, m: numbers of features


        '''
        self.Cl=Cl
        self.Cu=Cu
        self.kernel = kernel
        self.gamma=gamma
        self.clf=svm.SVC(C=self.Cl,kernel=kernel,gamma=self.gamma,probability=True)
        self.Yresult=None
        self.X2=X2

    def fit(self, X1, Y1):
        '''
        Train TSVM by X1, Y1, X2(X2 is passed on init)
        Parameters
        ----------
        X1: Input data with labels
                np.array, shape:[n1, m], n1: numbers of samples with labels, m: numbers of features
        Y1: labels of X1
                np.array, shape:[n1, ], n1: numbers of samples with labels


        '''

        t=time.time()
        X2=self.X2
        Y1[Y1!=+1]=-1
        Y1 = np.expand_dims(Y1, 1)
        ratio=sum(1 for i in Y1 if i==+1)/len(X1)
        num_plus=int(len(X2)*ratio) #number of positive example as describe in joachims svm

        N = len(X1) + len(X2)

        sample_weight = np.zeros(N)
        sample_weight[:len(X1)] = self.Cl


        self.clf.fit(X1, Y1,sample_weight=sample_weight[:len(X1)])  #classify the num_plus examples with the highest value with +1, other -1
        Y2=np.full(shape=self.clf.predict(X2).shape,fill_value=-1)
        Y2_d = self.clf.decision_function(X2)
        index=Y2_d.argsort()[-num_plus:][::-1]
        for item in index:
            Y2[item]=+1


        #INIT CMINUS E CLUS
        #C_minus=.00001
        C_minus=.00001
        C_plus=.00001*(num_plus/(len(X2)-num_plus))
        for i in range(len(Y2)):
            if(Y2[i]==+1):
                sample_weight[len(X1)+i]=C_plus
            else:
                sample_weight[len(X1)+i]=C_minus


        Y2 = np.expand_dims(Y2, 1)
        X3 = np.vstack((X1, X2))
        Y3 = np.vstack((Y1, Y2))
        k=0


        while (C_minus<self.Cu or C_plus<self.Cu): #LOOP 1
            self.clf.fit(X3, Y3, sample_weight=sample_weight)
            Y3 = Y3.reshape(-1)
            #slack=Y3*(self.clf.decision_function(X3))
            slack = Y3*self.clf.decision_function(X3)
            idx=np.argwhere(slack<1)
            eslackD=np.zeros(shape=slack.shape)
            for index in idx:
                eslackD[index]=1-slack[index]
            eslack2=np.zeros(shape=Y2.shape)
            eslack=eslackD[:len(X1)] #EPSILON OF LABELLED DATA
            eslack2=eslackD[len(X1):] # EPSILON FOR UNLABELED DATA


            condition=self.checkCondition(Y2,eslack2) #CONDITION OF LOOP
            l=0
            while(condition): #LOOP 2
                l+=1

                i,j=self.getIndexCondition(Y2,eslack2)  #TAKE A POSITIVE AND NEGATIVE SET
                #print("Switching at loop "+str(k)+"."+str(l)+"     index: "+str(i)+" "+str(j))
                #print("Switching values: "+str(eslack2[i])+" "+str(eslack2[j]))
                Y2[i]=Y2[i]*-1 #SWITCHING EXAMPLE
                Y2[j]= Y2[j]*-1

                sample_weight[len(X1)+i],sample_weight[len(X1)+j]=sample_weight[len(X1)+j],sample_weight[len(X1)+i] #UPDATE THE WEIGHT

                Y3=np.concatenate((Y1,Y2),axis=0)
                self.clf.fit(X3, Y3, sample_weight=sample_weight) #TRAINING WITH NEW LABELLING
                Y3 = Y3.reshape(-1)
                #slack =Y3*(self.clf.decision_function(X3))
                slack = Y3*self.clf.decision_function(X3)
                idx = np.argwhere(slack < 1)
                eslackD = np.zeros(shape=slack.shape)

                for index in idx:
                    eslackD[index] = 1 - slack[index]

                eslack = eslackD[:len(X1)]
                eslack2 = np.zeros(shape=Y2.shape)
                eslack2 = eslackD[len(X1):]
                condition = self.checkCondition(Y2, eslack2)
            k+=1
            #print(eslack2)
            C_minus=min(2*C_minus,self.Cu)
            C_plus=min(2*C_plus,self.Cu)
            #print("Loop "+str(k)+" Ctest="+str(self.Cu)+"   Cplus="+str(C_plus)+"   Cminus="+str(C_minus))

            for i in range(len(Y2)):
                if (Y2[i] == 1):
                    sample_weight[len(X1)+i] = C_plus
                else:
                    sample_weight[len(X1)+i] = C_minus

        self.Yresult=Y2
        Y3 = np.concatenate((Y1, Y2), axis=0)
        Y3=Y3.reshape(-1)
        end=time.time()
        print("The training finish in  "+str(end-t)+"  seconds")
        return self

    def checkCondition(self,Y,slack):
        '''
        Check condition of the loop 2
        Parameters
        ----------

        Y: labels of X2
                np.array, shape:[n1, ], n1: numbers of samples with semi-labels

        slack: slack variable for unlabeled set
                np.array, shape:[n1, ], n1: numbers of with semi-labels


        '''
        condition=False
        M=len(Y)
        for i in range(M):
            for j in range(M):
                if((Y[i]!=Y[j]) and (slack[i]>0) and (slack[j]>0) and ((slack[i]+slack[j])>2.001)):
                    condition=True
                    return condition
        return condition

    def getIndexCondition(self,Y,slack):
        '''
        Get index that satisfies condition of loop 2
        Parameters
        ----------

        Y: labels of X2
                np.array, shape:[n1, ], n1: numbers of samples with semi-labels

        slack: slack variable for unlabeled set
                np.array, shape:[n1, ], n1: numbers of with semi-labels


        '''
        M=len(Y)
        for i in range(M):
            for j in range(M):
                if(Y[i]!=Y[j] and slack[i]>0 and slack[j]>0 and (slack[i]+slack[j]>2.001)):
                    return i,j

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self):
        return self.clf.predict_proba()

    def decision_function(self, X):
        return self.clf.decision_function(X)

    def getResult(self):
        return self.Yresult

if __name__ == '__main__':
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score,roc_curve,f1_score

    X,y = load_breast_cancer(return_X_y=True)
    y[y==0]=-1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.985)
    print(X_train.shape,X_test.shape)
    gammas=[0.001,0.01,0.2,0.4,0.8,1.5,3]
    cls=[1,2,5,10,20]
    c=1
    g=0.0001
    clf1 = svm.SVC(C=c,kernel="linear",gamma=g)
    clf1.fit(X_train, y_train)

    y_svm = clf1.predict(X_test)
    f2 = accuracy_score(y_true=y_test, y_pred=y_svm)
    print("ACCURACY SVM " + str(f2))
    print("F1 SVM  ",f1_score(y_true=y_test,y_pred=y_svm))
    print("                                ")
    #print(clf1.coef_[0])
    clf=TransductiveSVM(kernel="linear",Cl=c,Cu=0.5,X2=X_test,gamma=g)
    clf.fit(X_train,y_train)
    y_predicted=clf.predict(X_test)
    f = accuracy_score(y_true=y_test, y_pred=y_predicted)
    print("ACCURACY TSVM " + str(f))
    print("F1 TSVM " + str(f1_score(y_true=y_test,y_pred=y_predicted)))
    #print(clf.clf.coef_[0])

    #print(y_svm)





