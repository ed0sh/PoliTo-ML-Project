import numpy
import util
from Classifiers.ClassifiersInterface import ClassifiersInterface
import scipy.special


class MultivariateGaussianClass(ClassifiersInterface):
    def __init__(self, DTR: numpy.array, LTR: numpy.array, prior: float):
        self.DTR = DTR
        self.LTR = LTR
        self.prior = prior
        self.nClasses = numpy.unique(LTR).shape[0]
        self.nSamples = DTR.shape[0]
        self.hCls = {}
        self.trained = False

    def classify(self, DTE: numpy.array):
        if not self.trained:
            raise RuntimeError('Classifier is not trained yet')

        priors = util.vcol(numpy.array([self.prior, 1 - self.prior]))
        logPrior = numpy.log(priors)
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
        return P

    def train(self):
        super().train()

    def update_dataset(self, DTR: numpy.array, LTR: numpy.array):
        self.DTR = DTR
        self.LTR = LTR
        self.nClasses = numpy.unique(LTR).shape[0]
        self.nSamples = DTR.shape[0]
        self.hCls = {}
        self.trained = False
