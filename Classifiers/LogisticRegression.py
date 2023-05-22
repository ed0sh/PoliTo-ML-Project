import numpy

import util
import scipy.special

from Classifiers.ClassifiersInterface import ClassifiersInterface


class LogRegClass(ClassifiersInterface):
    def __init__(self, DTR: numpy.array, LTR: numpy.array, l=1e-3):
        self.b = None
        self.w = None
        self.DTR = DTR
        self.ZTR = LTR * 2.0 - 1
        self.lam = l
        self.nSamples = DTR.shape[0]
        self.trained = False

    def logreg_obj(self, v):
        # Compute and return the objective function value. You can retrieve all required information from self.DTR, self.LTR, self.l
        w = util.vcol(v[0: self.nSamples])
        b = v[-1]
        scores = numpy.dot(w.T, self.DTR) + b
        loss_per_sample = numpy.logaddexp(0, -self.ZTR * scores)
        loss = 0.5 * self.lam * numpy.linalg.norm(w) ** 2 + loss_per_sample.mean()
        return loss

    def train(self):
        x0 = numpy.zeros(self.DTR.shape[0] + 1)
        xOpt, fOpt, d = scipy.optimize.fmin_l_bfgs_b(self.logreg_obj, x0=x0, approx_grad=True)
        self.w, self.b = util.vcol(xOpt[0:self.DTR.shape[0]]), xOpt[-1]
        self.trained = True

    def classify(self, DTE: numpy.array) -> numpy.array:
        if not self.trained:
            raise RuntimeError('Classifier is not trained yet')
        Score = numpy.dot(self.w.T, DTE) + self.b
        PLabel = (Score > 0).astype(int)
        return PLabel

    def evaluate(self, DTE: numpy.array, LTE: numpy.array) -> object:
        if not self.trained:
            raise RuntimeError('Classifier is not trained yet')
        Score = numpy.dot(self.w.T, DTE) + self.b
        PLabel = (Score > 0).astype(int)
        Error = ((LTE != PLabel).astype(int).sum() / DTE.shape[1]) * 100
        return Error, PLabel

    def confusion_matrix(self, DTE: numpy.array, LTE: numpy.array):
        _, PLabel = self.evaluate(DTE, LTE)
        return util.confusion_matrix(LTE, PLabel)
    @staticmethod
    def optimize_lambda(DTR: numpy.array , LTR: numpy.array , workPoint: util.WorkPoint) -> 'LogRegClass' :
        minDCF , selectedLambda = 1 , None
        for lam in [10 ** x for x in range(-8, 3)]:
            logReg = LogRegClass(DTR, LTR, lam)
            logReg.train()
            PLabels = logReg.classify(DTR)
            _, DCF = util.evaluate(PLabels, LTR, workPoint)
            if(DCF < minDCF):
                minDCF = DCF
                selectedLambda = lam
        return LogRegClass(DTR, LTR, selectedLambda)
