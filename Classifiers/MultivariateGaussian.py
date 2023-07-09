import numpy
import scipy.special

from Classifiers.ClassifiersInterface import ClassifiersInterface
from Utils import Util


class MultivariateGaussianClass(ClassifiersInterface):
    def __init__(self, DTR: numpy.array, LTR: numpy.array, prior: float):
        self.DTR = DTR
        self.LTR = LTR
        self.prior = prior
        self.nClasses = numpy.unique(LTR).shape[0]
        self.nFeatures = DTR.shape[0]
        self.hCls = {}
        self.trained = False

    def classify(self, DTE: numpy.array):
        if not self.trained:
            raise RuntimeError('Classifier is not trained yet')

        priors = Util.vcol(numpy.array([1 - self.prior, self.prior]))
        logPrior = numpy.log(priors)
        classes = numpy.unique(self.LTR)
        S = []
        for hyp in classes:
            C, mu = self.hCls[hyp]
            fcond = Util.logpdf_GAU_ND(DTE, mu, C)
            S.append(Util.vrow(fcond))

        S = numpy.vstack(S)

        SJoint = S + logPrior  # S is the logJoint
        logP = SJoint - Util.vrow(scipy.special.logsumexp(SJoint, 0))
        # logsumexp does in a numerical stable way the sum and the exp for marginal

        if classes.size == 2:
            self.scores = logP[1] - logP[0]

        P = numpy.exp(logP)

        predicted = numpy.argmax(P, axis=0)
        return predicted

    def train(self):
        super().train()

    def update_dataset(self, DTR: numpy.array, LTR: numpy.array):
        self.DTR = DTR
        self.LTR = LTR
        self.nClasses = numpy.unique(LTR).shape[0]
        self.nFeatures = DTR.shape[0]
        self.hCls = {}
        self.trained = False

    def get_scores(self) -> numpy.array:
        return super().get_scores()