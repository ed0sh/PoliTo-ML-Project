import numpy
import scipy.linalg

from Utils import Util


def PCA(D: numpy.array, m: int):
    C, _ = Util.dataCovarianceMatrix(D)
    U, Sigma, _ = numpy.linalg.svd(C)
    P = U[:, 0:m]

    expl_variance_fraction = Sigma[0:m].sum() / Sigma.sum()

    return numpy.dot(P.T, D), expl_variance_fraction


def LDA(D: numpy.array, L: numpy.array, m: int):
    if m > len(numpy.unique(L)) - 1:
        m = len(numpy.unique(L)) - 1

    Sw = Util.within_classes_covariance_matrix(D, L)
    Sb = Util.between_classes_covariance_matrix(D, L)
    W_g = Util.generalized_eig_problem(Sb, Sw, m)
    W_jd = Util.joint_diagonalizaton(Sb, Sw, m)

    sv = numpy.linalg.svd(numpy.hstack([W_jd, W_g]))[1]
    if len(sv[sv == 0]) > m:
        print("error in LDA")

    return numpy.dot(W_jd.T, D)


def Z_Score(D: numpy.array):
    for i in range(D.shape[0]):
        D[i] = (D[i] - D[i].mean()) / D[i].std()
    return D


