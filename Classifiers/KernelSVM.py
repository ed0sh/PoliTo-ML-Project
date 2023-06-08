import numpy

import util
import scipy.special


class KernelSVM:
    def __init__(self, DTR: numpy.array, LTR: numpy.array, C: float, K: int, d=2, c=0, gamma=1, kernel_ype="poly"):
        self.DTR = DTR
        self.LTR = LTR
        self.C = C
        self.K = K
        self.d = d
        self.c = c
        self.gamma = gamma

        if kernel_ype == "poly":
            self.kernel = self.poly_kernel
        elif kernel_ype == "rbf":
            self.kernel = self.RBF_kernel
        else:
            raise RuntimeError

        self.ZTR = LTR * 2.0 - 1
        self.nFeatures = DTR.shape[0]
        self.nSamples = DTR.shape[1]

        self.Hh = None
        self.alpha = None

    def RBF_kernel(self, X1, X2):
        slack_variable = self.K ** 2
        return numpy.exp(-self.gamma * (
                util.vcol(scipy.linalg.norm(X1, axis=0) ** 2)
                + util.vrow(scipy.linalg.norm(X2, axis=0) ** 2)
                - 2 * numpy.dot(X1.T, X2)
        )) + slack_variable

    def poly_kernel(self, X1, X2):
        slack_variable = self.K ** 2
        return ((numpy.dot(X1.T, X2) + self.c) ** self.d) + slack_variable

    def compute_Hh(self):
        Lh = numpy.dot(util.vcol(self.ZTR), util.vrow(self.ZTR))
        self.Hh = Lh * self.kernel(self.DTR, self.DTR)

    def dual_obj(self, alpha: numpy.array):
        alpha = util.vcol(alpha)
        loss = 0.5 * numpy.dot(alpha.T, numpy.dot(self.Hh, alpha)) - numpy.dot(alpha.T, numpy.ones((self.nSamples, 1)))
        loss_grad = numpy.dot(self.Hh, alpha) - numpy.ones((self.nSamples, 1))

        return loss.ravel()[0], loss_grad.ravel()

    def train(self):
        alpha = numpy.zeros(self.nSamples)
        bounds = [(0, self.C) for _ in range(self.nSamples)]
        alphaOpt, _, _ = scipy.optimize.fmin_l_bfgs_b(self.dual_obj, x0=alpha, bounds=bounds, factr=1.0)
        self.alpha = alphaOpt

        return alphaOpt

    def classify(self, DTE):
        S = (util.vcol(self.alpha * self.ZTR) * self.kernel(self.DTR, DTE)).sum(axis=0)
        predicted = (S > 0).astype(int)

        return predicted
