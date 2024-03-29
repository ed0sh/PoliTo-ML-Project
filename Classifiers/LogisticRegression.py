import numpy
import scipy.special

from Classifiers.ClassifiersInterface import ClassifiersInterface
from Utils import Util


class LogRegClass(ClassifiersInterface):
    def __init__(self, DTR: numpy.array, LTR: numpy.array, l=1e-3, prior=None):
        self.b = None
        self.w = None
        self.DTR = DTR
        self.LTR = LTR
        self.ZTR = LTR * 2.0 - 1
        self.lam = l
        self.nFeatures = DTR.shape[0]
        self.trained = False
        self.quadratic = False
        self.prior = prior

    def logreg_obj(self, v):
        # Compute and return the objective function value. You can retrieve all required information from self.DTR, self.LTR, self.l
        w = Util.vcol(v[0: self.nFeatures])
        b = v[-1]
        scores = numpy.dot(w.T, self.DTR) + b
        loss_per_sample = numpy.logaddexp(0, -self.ZTR * scores)
        loss = 0.5 * self.lam * numpy.linalg.norm(w) ** 2 + loss_per_sample.mean()
        return loss

    def prior_weighted_logreg_obj(self, v):
        # Compute and return the objective function value. You can retrieve all required information from self.DTR, self.LTR, self.l
        w = Util.vcol(v[0: self.nFeatures])
        b = v[-1]
        scores = numpy.dot(w.T, self.DTR) + b
        t_scores = scores[:, self.LTR == 1] + numpy.log(self.prior / (1 - self.prior))
        nt_scores = scores[:, self.LTR == 0] + numpy.log(self.prior / (1 - self.prior))

        t_loss_per_sample = self.prior * numpy.logaddexp(0, -t_scores)
        nt_loss_per_sample = (1 - self.prior) * numpy.logaddexp(0, nt_scores)

        loss = 0.5 * self.lam * numpy.linalg.norm(w) ** 2 + t_loss_per_sample.mean() + nt_loss_per_sample.mean()
        return loss

    def train(self):
        x0 = numpy.zeros(self.DTR.shape[0] + 1)
        if self.prior is None:
            xOpt, fOpt, d = scipy.optimize.fmin_l_bfgs_b(self.logreg_obj, x0=x0, approx_grad=True)
        else:
            xOpt, fOpt, d = scipy.optimize.fmin_l_bfgs_b(self.prior_weighted_logreg_obj, x0=x0, approx_grad=True)
        self.w, self.b = Util.vcol(xOpt[0:self.nFeatures]), xOpt[-1]
        self.trained = True

    def classify(self, DTE: numpy.array, workpoint: Util.WorkPoint) -> numpy.array:
        if not self.trained:
            raise RuntimeError('Classifier is not trained yet')
        Score = numpy.dot(self.w.T, DTE) + self.b
        self.scores = Score.ravel()

        t = - numpy.log(workpoint.effective_prior() / (1 - workpoint.effective_prior()))
        PLabel = (Score > t).astype(int).ravel()
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
    def create_with_optimized_lambda(DTR: numpy.array, LTR: numpy.array, workPoint: Util.WorkPoint, prior=None) -> 'LogRegClass':

        selectedLambda = 1e1
        lambdas = [1e1, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
        bestMinDCF = 10

        for lam in lambdas:
            logRegObj = LogRegClass(DTR, LTR, lam, prior)
            _, _, minDCF, _, _ = Util.k_folds(DTR, LTR, 5, logRegObj, workPoint)

            if minDCF < bestMinDCF:
                selectedLambda = lam
                bestMinDCF = minDCF

        return LogRegClass(DTR, LTR, selectedLambda, prior)

    def optimize_lambda_inplace(self, workPoint: Util.WorkPoint):

        selectedLambda = 1e1
        lambdas = [1e1, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
        bestMinDCF = 10

        for lam in lambdas:
            logRegObj = LogRegClass(self.DTR, self.LTR, lam, self.prior)
            _, _, minDCF, _, _ = Util.k_folds(self.DTR, self.LTR, 5, logRegObj, workPoint)

            if minDCF < bestMinDCF:
                selectedLambda = lam
                bestMinDCF = minDCF

        self.lam = selectedLambda

    def feature_expansion_inplace(self):
        Phi = []
        for i in range(self.DTR.shape[1]):
            x = Util.vcol(self.DTR[:, i])
            phi = numpy.vstack([Util.vcol(numpy.dot(x, x.T)), x])
            Phi.append(phi)
        Phi = numpy.hstack(Phi)
        self.DTR = Phi
        self.quadratic = True
        self.nFeatures = self.DTR.shape[0]
