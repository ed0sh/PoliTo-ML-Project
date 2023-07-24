import numpy

from Utils import Util


class GMM:
    def __init__(self, DTR: numpy.array, alpha: float, psi: float, max_g: int, diagonal_cov: bool, tied_cov: bool):
        self.DTR = DTR
        self.alpha = alpha
        self.psi = psi
        self.max_g = max_g

        C, mu = Util.dataCovarianceMatrix(self.DTR)
        self.gmm = [(1.0, mu, C)]

        self.diagonal_cov = diagonal_cov
        self.tied_cov = tied_cov
        self.nFeatures = DTR.shape[0]
        self.trained = False

    def logdens(self, DTE: numpy.array):
        if not self.trained:
            raise RuntimeError('Classifier is not trained yet')

        return Util.logpdf_GMM(DTE, self.gmm)

    def train(self):
        self.gmm, _, _ = Util.LBG(self.DTR, self.gmm, self.alpha, self.max_g, self.psi, self.diagonal_cov, self.tied_cov)
        self.trained = True
