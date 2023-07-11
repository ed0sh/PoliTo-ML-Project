import numpy
import scipy.special

from Classifiers.ClassifiersInterface import ClassifiersInterface
from Utils import Util


class LogRegClass(ClassifiersInterface):
    def __init__(self, DTR: numpy.array, LTR: numpy.array, l=1e-3):
        self.b = None
        self.w = None
        self.DTR = DTR
        self.LTR = LTR
        self.ZTR = LTR * 2.0 - 1
        self.lam = l
        self.nFeatures = DTR.shape[0]
        self.trained = False

    def logreg_obj(self, v):
        # Compute and return the objective function value. You can retrieve all required information from self.DTR, self.LTR, self.l
        w = Util.vcol(v[0: self.nFeatures])
        b = v[-1]
        scores = numpy.dot(w.T, self.DTR) + b
        loss_per_sample = numpy.logaddexp(0, -self.ZTR * scores)
        loss = 0.5 * self.lam * numpy.linalg.norm(w) ** 2 + loss_per_sample.mean()
        return loss

    def train(self):
        x0 = numpy.zeros(self.DTR.shape[0] + 1)
        xOpt, fOpt, d = scipy.optimize.fmin_l_bfgs_b(self.logreg_obj, x0=x0, approx_grad=True)
        self.w, self.b = Util.vcol(xOpt[0:self.nFeatures]), xOpt[-1]
        self.trained = True

    def classify(self, DTE: numpy.array) -> numpy.array:
        if not self.trained:
            raise RuntimeError('Classifier is not trained yet')
        Score = numpy.dot(self.w.T, DTE) + self.b
        self.scores = Score.ravel()
        PLabel = (Score > 0).astype(int).ravel()
        return PLabel

    def update_dataset(self, DTR: numpy.array, LTR: numpy.array):
        self.b = None
        self.w = None
        self.DTR = DTR
        self.LTR = LTR
        self.ZTR = LTR * 2.0 - 1
        self.nFeatures = DTR.shape[0]
        self.trained = False

    @staticmethod
    def create_with_optimized_lambda(DTR: numpy.array, LTR: numpy.array, workPoint: Util.WorkPoint) -> 'LogRegClass':

        selectedLambda = 1e1
        lambdas = [1e1, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
        bestMinDCF = 10

        for lam in lambdas:
            logRegObj = LogRegClass(DTR, LTR, lam)
            _, _, minDCF = Util.k_folds(DTR, LTR, 5, logRegObj, workPoint)

            if minDCF < bestMinDCF:
                selectedLambda = lam
                bestMinDCF = minDCF

        return LogRegClass(DTR, LTR, selectedLambda)
