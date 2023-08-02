import numpy

from Classifiers.GMM import GMM
from Classifiers.ClassifiersInterface import ClassifiersInterface
from Utils import Util


class GMMClassifier(ClassifiersInterface):
    def __init__(self,
                 DTR: numpy.array, LTR: numpy.array,
                 params_gmm_target: dict,
                 params_gmm_non_target: dict):

        self.DTR = DTR
        self.LTR = LTR
        self.params_gmm_target = params_gmm_target
        self.params_gmm_non_target = params_gmm_non_target
        self.trained = False
        self.scores = None
        self.gmms = []

        GMM_nt = GMM(self.DTR[:, self.LTR == 0],
                     self.params_gmm_non_target['alpha'],
                     self.params_gmm_non_target['psi'],
                     self.params_gmm_non_target['max_g'],
                     self.params_gmm_non_target['diagonal'],
                     self.params_gmm_non_target['tied'])
        GMM_t = GMM(self.DTR[:, self.LTR == 1],
                    self.params_gmm_target['alpha'],
                    self.params_gmm_target['psi'],
                    self.params_gmm_target['max_g'],
                    self.params_gmm_target['diagonal'],
                    self.params_gmm_target['tied'])
        self.gmms = [GMM_nt, GMM_t]

    def classify(self, DTE: numpy.array, workpoint: Util.WorkPoint):
        if not self.trained:
            raise RuntimeError('Classifier is not trained yet')

        log_dens = []
        for GMM_c in self.gmms:
            log_dens.append(GMM_c.logdens(DTE))
        log_dens = numpy.vstack(log_dens)

        if len(self.gmms) == 2:
            self.scores = log_dens[1] - log_dens[0]

        t = - numpy.log(workpoint.effective_prior() / (1 - workpoint.effective_prior()))
        predicted = (self.scores > t).astype(int)
        return predicted

    def train(self):
        for GMM_c in self.gmms:
            GMM_c.train()

        self.trained = True

    def update_dataset(self, DTR: numpy.array, LTR: numpy.array):
        self.DTR = DTR
        self.LTR = LTR

        GMM_nt = GMM(self.DTR[:, self.LTR == 0],
                     self.params_gmm_non_target['alpha'],
                     self.params_gmm_non_target['psi'],
                     self.params_gmm_non_target['max_g'],
                     self.params_gmm_non_target['diagonal'],
                     self.params_gmm_non_target['tied'])
        GMM_t = GMM(self.DTR[:, self.LTR == 1],
                    self.params_gmm_target['alpha'],
                    self.params_gmm_target['psi'],
                    self.params_gmm_target['max_g'],
                    self.params_gmm_target['diagonal'],
                    self.params_gmm_target['tied'])
        self.gmms = [GMM_nt, GMM_t]

        self.trained = False

    def __str__(self):
        return "GMM Classifier"
