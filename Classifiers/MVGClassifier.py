import numpy
from Classifiers.MultivariateGaussian import MultivariateGaussianClass
from Utils import Util


class MVGClassifier(MultivariateGaussianClass):
    def __init__(self, DTR: numpy.array, LTR: numpy.array, prior: float):
        super().__init__(DTR, LTR, prior)

    def classify(self, DTE: numpy.array, workpoint: Util.WorkPoint):
        return super().classify(DTE, workpoint)

    def train(self):
        for label in numpy.unique(self.LTR):
            DCLS = self.DTR[:, self.LTR == label]
            self.hCls[label] = Util.dataCovarianceMatrix(DCLS)

        self.trained = True
