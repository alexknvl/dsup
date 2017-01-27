from abc import *
#from sklearn import svm
import numpy as np
import scipy as sp
from scipy.sparse import *
from LR_sparse import LR, LR_XR, XR2
#from LR import LR, LR_XR, XR2
from Vocab import *


class Classifier:
    __metaclass__ = ABCMeta
    def __init__(self, target_expectation=0.5):
        pass
    @abstractmethod
    def Train(self, jsonDataset):
        pass
    @abstractmethod
    def Predict(self, jsonInstance):
        pass
    @abstractmethod
    def PrintWeights(self, outFile):
        pass

# class OcsvmClassifier(object):
#     def __init__(self, target_expectation=0.5):
#         self.model = None

#     def json2vocab(self, jsonInstance):
#         vocabd = {}
#         for k in jsonInstance.keys():
#             vocabd[self.vocab.GetID(k)] = jsonInstance[k]
#         return vocabd

#     def Train(self, jsonDataset, temp=None):
#         jsonDataset = [x for x in jsonDataset if x[1] == 1]     #Only positive instances
#         x,y = [d[0] for d in jsonDataset], [d[1] for d in jsonDataset]
#         self.vocab = Vocab()
#         x_vocabd = [self.json2vocab(d) for d in x]
#         problem = svm_problem(y,x_vocabd)
#         param = svm_parameter('-s 2 -t 0 -c 4')
#         self.model = svm_train(problem, param)

#     def Predict(self, jsonInstance):
#         p_label, p_acc, p_val = svm_predict([1], [self.json2vocab(jsonInstance)], self.model, '-b 0')
#         print (p_label, p_val)
#         return p_val[0][0]

#     def PrintWeights(self, outFile):
#         pass

# class SVMclassifier(object):
#     def __init__(self):
#         self.model = None

#     def json2vocab(self, jsonInstance):
#         vocabd = {}
#         for k in jsonInstance.keys():
#             vocabd[self.vocab.GetID(k)] = jsonInstance[k]
#         return vocabd

#     def json2vector(self, jsonInstance):
#         result = np.zeros(self.vocab.GetVocabSize())
#         for k in jsonInstance.keys():
#             if self.vocab.GetID(k) > 0:
#                 result[self.vocab.GetID(k)-1] = jsonInstance[k]
#         return result

#     def Train(self, jsonDataset, temp=None, l2=1.0):
#         x,y = [d[0] for d in jsonDataset], [d[1] for d in jsonDataset]
#         self.vocab = Vocab()
#         x_vocabd = [self.json2vocab(d) for d in x]
#         self.vocab.Lock()
#         X_matrix = np.zeros((len(x_vocabd), self.vocab.GetVocabSize()))
#         for i in range(len(x_vocabd)):
#             for (j,v) in x_vocabd[i].items():
#                 X_matrix[i,j-1] = v

#         self.model = svm.LinearSVC()
#         self.model.fit(X_matrix,np.array(y))
# #        lr = LR(X_matrix,np.array(y), l2=l2)
# #        lr.Train()
# #        self.model = lr

#     def Predict(self, jsonInstance):
#         return self.model.predict(self.json2vector(jsonInstance))

#     #TODO:
#     def PrintWeights(self, outFile):
#         pass
#         #fOut = open(outFile, 'w')
#         #for i in np.argsort(-self.model.wStar):
#         #    fOut.write("%s\t%s\n" % (self.vocab.GetWord(i+1), self.model.wStar[i]))

class LRclassifier(object):
    def __init__(self, target_expectation=0.5):
        self.model = None

    def json2vocab(self, jsonInstance):
        vocabd = {}
        for k in jsonInstance.keys():
            vocabd[self.vocab.GetID(k)] = jsonInstance[k]
        return vocabd

    def json2vector(self, jsonInstance):
        result = np.zeros(self.vocab.GetVocabSize())
        for k in jsonInstance.keys():
            if self.vocab.GetID(k) > 0:
                result[self.vocab.GetID(k)-1] = jsonInstance[k]
        return result

    def Train(self, jsonDataset, temp=None, l2=1.0):
        x,y = [d[0] for d in jsonDataset], [d[1] for d in jsonDataset]
        self.vocab = Vocab()
        x_vocabd = [self.json2vocab(d) for d in x]
        self.vocab.Lock()
        X_matrix = np.zeros((len(x_vocabd), self.vocab.GetVocabSize()))
        for i in range(len(x_vocabd)):
            for (j,v) in x_vocabd[i].items():
                X_matrix[i,j-1] = v
        lr = LR(X_matrix,np.array(y), l2=l2)
        lr.Train()
        self.model = lr

    def Predict(self, jsonInstance):
        return self.model.Predict(self.json2vector(jsonInstance))

    def PrintWeights(self, outFile):
        fOut = open(outFile, 'w')
        for i in np.argsort(-self.model.wStar):
            fOut.write("%s\t%s\n" % (self.vocab.GetWord(i+1), self.model.wStar[i]))


def densify(x, N):
    result = np.zeros(N)
    for i in x:
        result[i] = 1
    return result


class LRclassifierV2(object):
    def __init__(self):
        self.model = None

    def Prepare(self, data, vocabulary, index):
        self.vocabulary = vocabulary
        self.index = index
        X_matrix = lil_matrix((len(data), len(self.vocabulary)))
        Y = np.zeros(len(data))

        print "populating lil_matrix"
        for i, sample in enumerate(data):
            Y[i] = sample[1]
            for j in sample[0]:
                X_matrix[i, j] = 1
        print "done populating lil_matrix"

        print "tocsr"
        X_matrix = X_matrix.tocsr()
        print "done tocsr"

        self.X = X_matrix.tocsr()
        self.Y = Y

    def Train(self, l2=1.0):
        lr = LR(self.X, self.Y, l2=l2)
        lr.Train()
        self.model = lr

    def Predict(self, data):
        return self.model.Predict(densify(data, len(self.vocabulary)))

    def PrintWeights(self, outFile):
        fOut = open(outFile, 'w')
        for i in np.argsort(-self.model.wStar):
            fOut.write("%s\t%s\n" % (self.vocabulary[i], self.model.wStar[i]))


class LR_XRclassifierV2(object):
    def __init__(self):
        self.model = None

    def Prepare(self, data, vocabulary, index):
        self.vocabulary = vocabulary
        self.index = index
        X_matrix = lil_matrix((len(data), len(self.vocabulary)))
        Y = np.zeros(len(data))

        print "populating lil_matrix"
        for i, sample in enumerate(data):
            Y[i] = sample[1]
            for j in sample[0]:
                X_matrix[i, j] = 1
        print "done populating lil_matrix"

        print "tocsr"
        X_matrix = X_matrix.tocsr()
        print "done tocsr"

        print "creating LR_XR"
        #lr = LR_XR(X_matrix[Y == 1,:].tocsr(), Y[Y==1], X_matrix[Y==-1].tocsr(), p_ex=0.4, temp=temp, xr=10.0, l2=l2)
        #lr = LR_XR(X_matrix[Y == 1,:], Y[Y==1], X_matrix[Y==-1, :], p_ex=0.4, temp=temp, xr=10.0, l2=l2)
        pos_indices = (Y ==  1).nonzero()[0]    #positive (need to do this for sparse matrices)
        unl_indices = (Y == -1).nonzero()[0]    #unlabeled
        #lr = LR_XR(X_matrix[pos_indices,:], Y[Y==1], X_matrix[unl_indices, :], p_ex=0.5, temp=temp, xr=10.0, l2=l2)
        self.X = X_matrix[pos_indices,:]
        self.Y = Y[Y==1]
        self.U = X_matrix[unl_indices, :]

    def Train(self, p_ex=0.5, temp=1.0, l2=1.0, xr=10.0):
        lr = LR_XR(self.X, self.Y, self.U, p_ex=p_ex, temp=temp, xr=xr, l2=l2)
        print "done creating LR_XR"
        lr.Train()

        print lr.nll
        print lr.status

        self.model = lr

    def Predict(self, data):
        return self.model.Predict(densify(data, len(self.vocabulary)))

    def PrintWeights(self, outFile):
        fOut = open(outFile, 'w')
        for i in np.argsort(-self.model.wStar):
            fOut.write("%s\t%s\n" % (self.vocabulary[i], self.model.wStar[i]))

class LR_XRclassifier(object):
    def __init__(self, target_expectation=0.5):
        self.model = None
        self.p_ex = target_expectation

    def json2vocab(self, jsonInstance):
        vocabd = {}
        for k in jsonInstance.keys():
            vocabd[self.vocab.GetID(k)-1] = jsonInstance[k]
        return vocabd

    def json2vector(self, jsonInstance):
        result = np.zeros(self.vocab.GetVocabSize())
        for k in jsonInstance.keys():
            if self.vocab.GetID(k) > 0:
                result[self.vocab.GetID(k)-1] = jsonInstance[k]
        return result

    def Train(self, jsonDataset, temp=1.0, l2=1.0):
        x,y = [d[0] for d in jsonDataset], [d[1] for d in jsonDataset]
        self.vocab = Vocab()
        x_vocabd = [self.json2vocab(d) for d in x]
        self.vocab.Lock()
        #X_matrix = np.zeros((len(x_vocabd), self.vocab.GetVocabSize()))
        X_matrix = lil_matrix((len(x_vocabd), self.vocab.GetVocabSize()))

        print "populating lil_matrix"
        for i in range(len(x_vocabd)):
            for (j,v) in x_vocabd[i].items():
                X_matrix[i,j] = v
        print "done populating lil_matrix"

        Y = np.array(y)

        print "tocsr"
        X_matrix = X_matrix.tocsr()
        print "done tocsr"

        print "creating LR_XR"
        #lr = LR_XR(X_matrix[Y == 1,:].tocsr(), Y[Y==1], X_matrix[Y==-1].tocsr(), p_ex=0.4, temp=temp, xr=10.0, l2=l2)
        #lr = LR_XR(X_matrix[Y == 1,:], Y[Y==1], X_matrix[Y==-1, :], p_ex=0.4, temp=temp, xr=10.0, l2=l2)
        pos_indices = (Y ==  1).nonzero()[0]    #positive (need to do this for sparse matrices)
        unl_indices = (Y == -1).nonzero()[0]    #unlabeled
        #lr = LR_XR(X_matrix[pos_indices,:], Y[Y==1], X_matrix[unl_indices, :], p_ex=0.5, temp=temp, xr=10.0, l2=l2)
        lr = LR_XR(X_matrix[pos_indices,:], Y[Y==1], X_matrix[unl_indices, :], p_ex=self.p_ex, temp=temp, xr=10.0, l2=l2)
        print "done creating LR_XR"
        lr.Train()

        print lr.nll
        print lr.status

        self.model = lr

    def Predict(self, jsonInstance):
        return self.model.Predict(self.json2vector(jsonInstance))

    def PrintWeights(self, outFile):
        fOut = open(outFile, 'w')
        for i in np.argsort(-self.model.wStar):
            fOut.write("%s\t%s\n" % (self.vocab.GetWord(i+1), self.model.wStar[i]))

class XR2classifier(object):
    def __init__(self, target_expectation=0.5):
        self.model = None

    def json2vocab(self, jsonInstance):
        vocabd = {}
        for k in jsonInstance.keys():
            vocabd[self.vocab.GetID(k)-1] = jsonInstance[k]
        return vocabd

    def json2vector(self, jsonInstance):
        result = np.zeros(self.vocab.GetVocabSize())
        for k in jsonInstance.keys():
            if self.vocab.GetID(k) > 0:
                result[self.vocab.GetID(k)-1] = jsonInstance[k]
        return result

    def Train(self, jsonDataset, temp=1.0, l2=1.0):
        x,y = [d[0] for d in jsonDataset], [d[1] for d in jsonDataset]
        self.vocab = Vocab()
        x_vocabd = [self.json2vocab(d) for d in x]
        self.vocab.Lock()
        X_matrix = np.zeros((len(x_vocabd), self.vocab.GetVocabSize()))
        for i in range(len(x_vocabd)):
            for (j,v) in x_vocabd[i].items():
                X_matrix[i,j] = v

        Y = np.array(y)

        lr = XR2(X_matrix[Y == 1,:], X_matrix[Y==-1], p_ex1=0.95, p_ex2=0.4, temp=1.0, l2=l2, xr1=1.0, xr2=1.0) #-> *
        #lr = XR2(X_matrix[Y == 1,:], X_matrix[Y==-1], p_ex1=0.95, p_ex2=0.56, temp=1.0, l2=1.0, xr1=10.0, xr2=200.0)
        lr.Train()

#        print "weights:%s" % list(lr.wStar)
        print lr.nll
        print lr.status

        self.model = lr

    def Predict(self, jsonInstance):
        return self.model.Predict(self.json2vector(jsonInstance))

    def PrintWeights(self, outFile):
        fOut = open(outFile, 'w')
        for i in np.argsort(-self.model.wStar):
            fOut.write("%s\t%s\n" % (self.vocab.GetWord(i+1), self.model.wStar[i]))

# Classifier.register(OcsvmClassifier)
Classifier.register(LRclassifier)
Classifier.register(LR_XRclassifier)
Classifier.register(XR2classifier)
