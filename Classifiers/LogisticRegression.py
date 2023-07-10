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
    def optimize_lambda(DTR: numpy.array, LTR: numpy.array, workPoint: Util.WorkPoint, tolerance: float = 1e-5,
                        num_iterations: int = 100, starting_lambda: float = 1e-1, offset: float = 1) -> 'LogRegClass':

        minDCF, selectedLambda, offset = 1, starting_lambda, offset
        prev_DCF = float('inf')

        while (num_iterations > 0):
            logRegObj = LogRegClass(DTR, LTR, selectedLambda)

            _, DCF, _ = Util.k_folds(DTR, LTR, 4, logRegObj, workPoint)

            # Calculate change derivative
            change = prev_DCF - DCF
            prev_DCF = DCF

            if numpy.abs(change) < tolerance:
                break

            if DCF < minDCF:
                minDCF = DCF

            if change > 0:
                selectedLambda = selectedLambda + offset
            else:
                offset = - offset / 2
                selectedLambda = selectedLambda + offset
            num_iterations -= 1

        return LogRegClass(DTR, LTR, selectedLambda)

    def optimize_lambda_inplace(self, workPoint: Util.WorkPoint, tolerance: float = 1e-5,
                        num_iterations: int = 100, starting_lambda: float = 1e-1, offset: float = 1):

        minDCF = 1
        selectedLambda = starting_lambda
        offset = offset
        prev_DCF = float('inf')

        while num_iterations > 0:
            logRegObj = LogRegClass(self.DTR, self.LTR, selectedLambda)
            _, DCF,_ = Util.k_folds(self.DTR, self.LTR, 5, logRegObj, workPoint)

            # Calculate change derivative
            change = prev_DCF - DCF
            prev_DCF = DCF

            if numpy.abs(offset) < tolerance:
                break

            if DCF < minDCF:
                minDCF = DCF

            if change > 0:
                selectedLambda = selectedLambda + offset
            else:
                offset = - offset * 9 / 10
                selectedLambda = selectedLambda + offset

            num_iterations -= 1

        self.lam = selectedLambda

    def feature_expansion_inplace(self):
        Phi = []
        for i in range(self.DTR.shape[1]):
            x = Util.vcol(self.DTR[:, i])
            phi = numpy.vstack([Util.vcol(numpy.dot(x, x.T)), x])
            Phi.append(phi)
        Phi = numpy.hstack(Phi)
        self.DTR = Phi
