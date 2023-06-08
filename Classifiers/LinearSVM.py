import numpy

import util
import scipy.special


class LinearSVM:
    def __init__(self, DTR: numpy.array, LTR: numpy.array, C: float, K: int):
        self.DTR = DTR
        self.LTR = LTR
        self.C = C
        self.K = K
        self.ZTR = LTR * 2.0 - 1
        self.nFeatures = DTR.shape[0]
        self.nSamples = DTR.shape[1]
        self.Hh = None
        self.wh = None

    def compute_Hh(self):
        Dh = numpy.vstack([self.DTR, self.K * numpy.ones(self.nSamples)])
        Gh = numpy.dot(Dh.T, Dh)
        Lh = numpy.dot(util.vcol(self.ZTR), util.vrow(self.ZTR))
        self.Hh = Lh * Gh

    def primal_obj(self, wh: numpy.array):
        Dh = numpy.vstack([self.DTR, self.K * numpy.ones(self.nSamples)])
        loss = (0.5 * scipy.linalg.norm(wh) ** 2) \
               + (self.C * (numpy.maximum(
                                numpy.zeros(self.nSamples),
                                1 - (self.ZTR * (numpy.dot(wh.T, Dh))))
                            ).sum()
                  )

        return loss

    def dual_obj(self, alpha: numpy.array):
        loss = 0.5 * numpy.dot(util.vrow(alpha), numpy.dot(self.Hh, util.vcol(alpha))) \
               - numpy.dot(util.vrow(alpha), numpy.ones((self.nSamples, 1)))
        loss_grad = numpy.dot(self.Hh, util.vcol(alpha)) - numpy.ones((self.nSamples, 1))

        return loss.ravel()[0], loss_grad.ravel()

    def train(self):
        alpha = numpy.zeros(self.nSamples)
        bounds = [(0, self.C) for _ in range(self.nSamples)]
        alphaOpt, fOpt, d = scipy.optimize.fmin_l_bfgs_b(self.dual_obj, x0=alpha, bounds=bounds, factr=1.0)
        Dh = numpy.vstack([self.DTR, self.K * numpy.ones(self.DTR.shape[1])])
        wh = ((util.vrow(alphaOpt) * util.vrow(self.ZTR)) * Dh).sum(axis=1)
        self.wh = wh

        return alphaOpt, wh

    def classify(self, DTE):
        Dth = numpy.vstack([DTE, self.K * numpy.ones(DTE.shape[1])])
        S = numpy.dot(self.wh.T, Dth)
        predicted = (S > 0).astype(int)

        return predicted
