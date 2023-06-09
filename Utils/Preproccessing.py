import numpy
from Utils import Util


def PCA(D: numpy.array, m: int):
    C, _ = Util.dataCovarianceMatrix(D)
    U, Sigma, _ = numpy.linalg.svd(C)
    P = U[:, 0:m]

    expl_variance_fraction = Sigma[0:m].sum() / Sigma.sum()

    return numpy.dot(P.T, D), expl_variance_fraction


def LDA(D: numpy.array):
    # TODO: In the project example the prof took it into account
    pass


def Z_Score(D: numpy.array):
    for i in range(D.shape[0]):
        D[i] = (D[i] - D[i].mean()) / D[i].std()
    return D

