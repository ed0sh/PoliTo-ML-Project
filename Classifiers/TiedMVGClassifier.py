import numpy

from Classifiers.MultivariateGaussian import MultivariateGaussianClass
from Utils import Util


class TiedMVGClassifier(MultivariateGaussianClass):
    def __init__(self, DTR: numpy.array, LTR: numpy.array, prior: float):
        super().__init__(DTR, LTR, prior)

    def classify(self, DTE: numpy.array, workpoint: Util.WorkPoint):
        return super().classify(DTE, workpoint)

    def train(self):
        Sw = Util.within_classes_covariance_matrix(self.DTR, self.LTR)

        for lab in numpy.unique(self.LTR):
            DCLS = self.DTR[:, self.LTR == lab]
            _, mu = Util.dataCovarianceMatrix(DCLS)
            self.hCls[lab] = (Sw, mu)

        self.trained = True
