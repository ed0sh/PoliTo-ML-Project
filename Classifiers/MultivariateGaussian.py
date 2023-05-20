import numpy
import util
from Classifiers.ClassifiersInterface import ClassifiersInterface
import scipy.special


class MultivariateGaussianClass(ClassifiersInterface):
    def __init__(self, DTR: numpy.array, LTR: numpy.array):
        self.DTR = DTR
        self.LTR = LTR
        self.nClasses = numpy.unique(LTR).shape[0]
        self.nSamples = DTR.shape[0]
        self.hCls = {}

    def classify(self, DTE: numpy.array):
        logPrior = numpy.log(util.vcol(numpy.ones(self.nClasses) / self.nClasses))
        S = []
        for hyp in numpy.unique(self.LTR):
            C, mu = self.hCls[hyp]
            fcond = util.logpdf_GAU_ND(DTE, mu, C)
            S.append(util.vrow(fcond))
        S = numpy.vstack(S)

        SJoint = S + logPrior  # S is the logJoint
        logP = SJoint - util.vrow(scipy.special.logsumexp(SJoint, 0))
        # logsumexp does in a numerical stable way the sum and the exp for marginal
        P = numpy.exp(logP)
        return SJoint, P

    def train(self):
        super().train()

