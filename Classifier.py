import numpy
import util
from util import logpdf_GAU_ND
import scipy.special


# MultivariateGaussianClassifier
def linearMVG(DTR, LTR, DTE):
    hCls = {}
    n = numpy.unique(LTR).shape[0]
    for lab in numpy.unique(LTR):
        DCLS = DTR[:, LTR == lab]
        hCls[lab] = util.dataCovarianceMatrix(DCLS)
    ### Classification
    prior = util.vcol(numpy.ones(n) / n)
    S = []
    for hyp in numpy.unique(LTR):
        C, mu = hCls[hyp]
        fcond = numpy.exp(logpdf_GAU_ND(DTE, mu, C))
        S.append(util.vrow(fcond))
    S = numpy.vstack(S)
    SJoint = S * prior
    P = SJoint / util.vrow(SJoint.sum(0))  # vrow(S.sum(0)) is the marginal

    return SJoint, P


def logMVG(DTR, LTR, DTE):
    hCls = {}
    n = numpy.unique(LTR).shape[0]
    for lab in numpy.unique(LTR):
        DCLS = DTR[:, LTR == lab]
        hCls[lab] = util.dataCovarianceMatrix(DCLS)
    # classification
    logprior = numpy.log(util.vcol(numpy.ones(n) / n))
    S = []
    for hyp in numpy.unique(LTR):
        C, mu = hCls[hyp]
        fcond = logpdf_GAU_ND(DTE, mu, C)
        S.append(util.vrow(fcond))
    S = numpy.vstack(S)
    SJoint = S + logprior  # S is the logJoint
    logP = SJoint - util.vrow(scipy.special.logsumexp(SJoint, 0))
    # logsumexp does in a numerical stable way the sum and the exp for marginal
    P = numpy.exp(logP)
    return SJoint, P


def logNaiveMVG(DTR, LTR, DTE):
    hCls = {}
    n = numpy.unique(LTR).shape[0]
    for lab in numpy.unique(LTR):
        DCLS = DTR[:, LTR == lab]
        C, mu = util.dataCovarianceMatrix(DCLS)
        ones = numpy.diag(numpy.ones(DCLS.shape[0]))
        hCls[lab] = (C * ones, mu)

    logprior = numpy.log(util.vcol(numpy.ones(n) / n))
    S = []
    for hyp in numpy.unique(LTR):
        C, mu = hCls[hyp]
        fcond = logpdf_GAU_ND(DTE, mu, C)
        S.append(util.vrow(fcond))
    S = numpy.vstack(S)
    SJoint = S + logprior  # S is the logJoint
    logP = SJoint - util.vrow(scipy.special.logsumexp(SJoint, 0))
    # logsumexp does in a numerical stable way the sum and the exp for marginal
    P = numpy.exp(logP)

    return SJoint, P


def logTiedMVG(DTR, LTR, DTE):
    hCls = {}
    n = numpy.unique(LTR).shape[0]
    Sw = numpy.zeros((DTR.shape[0], DTR.shape[0]))
    for i in range(0, n):
        Sw += util.within_class_covariance(DTR[:, LTR == i], DTR.size)

    for lab in numpy.unique(LTR):
        DCLS = DTR[:, LTR == lab]
        _, mu = util.dataCovarianceMatrix(DCLS)
        hCls[lab] = (Sw, mu)

    logprior = numpy.log(util.vcol(numpy.ones(n) / n))
    S = []
    for hyp in numpy.unique(LTR):
        C, mu = hCls[hyp]
        fcond = logpdf_GAU_ND(DTE, mu, C)
        S.append(util.vrow(fcond))
    S = numpy.vstack(S)
    SJoint = S + logprior  # S is the logJoint
    logP = SJoint - util.vrow(scipy.special.logsumexp(SJoint, 0))
    # logsumexp does in a numerical stable way the sum and the exp for marginal
    P = numpy.exp(logP)

    return SJoint, P