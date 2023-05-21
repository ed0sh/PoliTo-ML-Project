import numpy
import util
from Classifiers.MultivariateGaussian import MultivariateGaussianClass


class MVGClassifier(MultivariateGaussianClass):
    def __init__(self, DTR: numpy.array, LTR: numpy.array, prior: float):
        super().__init__(DTR, LTR, prior)

    def classify(self, DTE: numpy.array):
        return super().classify(DTE)

    def train(self):
        for label in numpy.unique(self.LTR):
            DCLS = self.DTR[:, self.LTR == label]
            self.hCls[label] = util.dataCovarianceMatrix(DCLS)

        self.trained = True
