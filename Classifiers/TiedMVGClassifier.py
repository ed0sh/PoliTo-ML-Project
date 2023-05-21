import numpy
import util
from Classifiers.MultivariateGaussian import MultivariateGaussianClass


class TiedMVGClassifier(MultivariateGaussianClass):
    def __init__(self, DTR: numpy.array, LTR: numpy.array, prior: float):
        super().__init__(DTR, LTR, prior)

    def classify(self, DTE: numpy.array):
        return super().classify(DTE)

    def train(self):
        Sw = numpy.zeros((self.nSamples, self.nSamples))
        for i in range(0, self.nClasses):
            Sw += util.within_class_covariance(self.DTR[:, self.LTR == i], self.DTR.size)

        for lab in numpy.unique(self.LTR):
            DCLS = self.DTR[:, self.LTR == lab]
            _, mu = util.dataCovarianceMatrix(DCLS)
            self.hCls[lab] = (Sw, mu)

        self.trained = True
