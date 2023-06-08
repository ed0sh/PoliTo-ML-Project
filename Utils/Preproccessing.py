import numpy
from Utils import Util


def PCA(D: numpy.array, m: int):
    C, _ = Util.dataCovarianceMatrix(D)
    U, _, _ = numpy.linalg.svd(C)
    P = U[:, 0:m]
    return numpy.dot(P.T, D)


def LDA(D: numpy.array):
    pass


def Z_Score(D: numpy.array):
    for i in range(D.shape[0]):
        D[i] = (D[i] - D[i].mean()) / D[i].std()
    return D

