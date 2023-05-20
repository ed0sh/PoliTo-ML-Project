import numpy

import util
import scipy.special


class LogRegClass:
    def __init__(self, DTR: numpy.array, LTR: numpy.array, l=1e-3):
        self.DTR = DTR
        self.ZTR = LTR * 2.0 - 1
        self.l = l
        self.nSamples = DTR.shape[0]

    def logreg_obj(self, v):
        # Compute and return the objective function value. You can retrieve all required information from self.DTR, self.LTR, self.l
        w = util.vcol(v[0: self.nSamples])
        b = v[-1]
        scores = numpy.dot(w.T, self.DTR) + b
        loss_per_sample = numpy.logaddexp(0, -self.ZTR * scores)
        loss = 0.5 * self.l * numpy.linalg.norm(w) ** 2 + loss_per_sample.mean()
        return loss

    def train(self):
        x0 = numpy.zeros(self.DTR.shape[0] + 1)
        xOpt, fOpt, d = scipy.optimize.fmin_l_bfgs_b(self.logreg_obj, x0=x0, approx_grad=True)
        w, b = util.vcol(xOpt[0:self.DTR.shape[0]]), xOpt[-1]
        return w, b

    def evaluate(self, DTE: numpy.array, LTE: numpy.array):
        w, b = self.train()
        Score = numpy.dot(w.T, DTE) + b
        PLabel = (Score > 0).astype(int)
        Error = ((LTE != PLabel).astype(int).sum() / DTE.shape[1]) * 100
        return Error, PLabel

    def confusion_matrix(self, DTE: numpy.array, LTE: numpy.array):
        _, PLabel = self.evaluate(DTE, LTE)
        return util.confusion_matrix(LTE, PLabel)
