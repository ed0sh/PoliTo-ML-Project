import numpy
import util
from Classifiers.MultivariateGaussian import MultivariateGaussianClass


class NaiveMVGClassifier(MultivariateGaussianClass):
    def __init__(self, DTR: numpy.array, LTR: numpy.array):
        super().__init__(DTR, LTR)

    def classify(self, DTE: numpy.array):
        return super().classify(DTE)

    def train(self):
        for lab in numpy.unique(self.LTR):
            DCLS = self.DTR[:, self.LTR == lab]
            C, mu = util.dataCovarianceMatrix(DCLS)
            ones = numpy.diag(numpy.ones(DCLS.shape[0]))
            self.hCls[lab] = (C * ones, mu)

        self.trained = True