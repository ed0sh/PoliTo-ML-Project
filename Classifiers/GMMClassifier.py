import numpy

from GMM import GMM
from Classifiers.ClassifiersInterface import ClassifiersInterface


class GMMClassifier(ClassifiersInterface):
    def __init__(self, DTR: numpy.array, LTR: numpy.array, alpha: float, psi: float, max_g: int, sigma_type: str):
        self.DTR = DTR
        self.LTR = LTR
        self.alpha = alpha
        self.psi = psi
        self.max_g = max_g
        self.sigma_type = sigma_type
        self.trained = False

        self.gmms = []
        for c in numpy.unique(self.LTR):
            GMM_c = GMM(self.DTR[:, self.LTR == c], self.alpha, self.psi, self.max_g, self.sigma_type)
            self.gmms.append(GMM_c)

    def classify(self, DTE: numpy.array):
        if not self.trained:
            raise RuntimeError('Classifier is not trained yet')

        log_dens = []
        for GMM_c in self.gmms:
            log_dens.append(GMM_c.logdens(DTE))
        log_dens = numpy.vstack(log_dens)

        predicted = numpy.argmax(log_dens, axis=0)
        return predicted

    def train(self):
        for GMM_c in self.gmms:
            GMM_c.train()

        self.trained = True

    def update_dataset(self, DTR: numpy.array, LTR: numpy.array):
        self.DTR = DTR
        self.LTR = LTR

        self.gmms = []
        for c in numpy.unique(self.LTR):
            GMM_c = GMM(self.DTR[:, self.LTR == c], self.alpha, self.psi, self.max_g, self.sigma_type)
            self.gmms.append(GMM_c)

        self.trained = False
