import numpy
import scipy.special

from Classifiers.ClassifiersInterface import ClassifiersInterface
from Utils import Util


class KernelSVM(ClassifiersInterface):
    def __init__(self, DTR: numpy.array, LTR: numpy.array, C: float, K: int, d=2, c=0, gamma=1, kernel_type="poly"):
        self.DTR = DTR
        self.LTR = LTR
        self.C = C
        self.balanced_C = None
        self.K = K
        self.d = d
        self.c = c
        self.gamma = gamma

        if kernel_type == "poly":
            self.kernel = self.poly_kernel
        elif kernel_type == "rbf":
            self.kernel = self.RBF_kernel
        else:
            raise RuntimeError

        self.ZTR = LTR * 2.0 - 1
        self.nFeatures = DTR.shape[0]
        self.nSamples = DTR.shape[1]

        self.Hh = None
        self.alpha = None
        self.trained = False

    def RBF_kernel(self, X1, X2):
        slack_variable = self.K ** 2
        return numpy.exp(-self.gamma * (
                Util.vcol(scipy.linalg.norm(X1, axis=0) ** 2)
                + Util.vrow(scipy.linalg.norm(X2, axis=0) ** 2)
                - 2 * numpy.dot(X1.T, X2)
        )) + slack_variable

    def poly_kernel(self, X1, X2):
        slack_variable = self.K ** 2
        return ((numpy.dot(X1.T, X2) + self.c) ** self.d) + slack_variable

    def compute_Hh(self):
        Lh = numpy.dot(Util.vcol(self.ZTR), Util.vrow(self.ZTR))
        self.Hh = Lh * self.kernel(self.DTR, self.DTR)

    def dual_obj(self, alpha: numpy.array):
        alpha = Util.vcol(alpha)
        loss = 0.5 * numpy.dot(alpha.T, numpy.dot(self.Hh, alpha)) - alpha.sum()
        loss_grad = numpy.dot(self.Hh, alpha) - numpy.ones(alpha.shape)

        return loss.ravel()[0], loss_grad.ravel()

    def rebalance(self, workpoint: Util.WorkPoint):
        n = float(self.DTR.shape[1])
        nT = float((self.LTR > 0).sum())

        emp_pi = nT / n
        balanced_C = self.C * numpy.array([workpoint.pi / emp_pi, (1 - workpoint.pi) / (1 - emp_pi)])
        self.balanced_C = [balanced_C[label] for label in self.LTR]

    def train(self):
        self.compute_Hh()
        alpha = numpy.zeros(self.nSamples)
        if self.balanced_C is not None:
            bounds = [(0, self.balanced_C[i]) for i in range(self.nSamples)]
        else:
            bounds = [(0, self.C) for _ in range(self.nSamples)]
        alphaOpt, _, _ = scipy.optimize.fmin_l_bfgs_b(self.dual_obj, x0=alpha, bounds=bounds, factr=1.0)
        self.alpha = alphaOpt
        self.trained = True

        return alphaOpt

    def classify(self, DTE):
        if not self.trained:
            raise RuntimeError('Classifier is not trained yet')

        S = (Util.vcol(self.alpha * self.ZTR) * self.kernel(self.DTR, DTE)).sum(axis=0)
        self.scores = S.ravel()
        predicted = (S > 0).astype(int)

        return predicted

    def update_dataset(self, DTR: numpy.array, LTR: numpy.array):
        self.DTR = DTR
        self.LTR = LTR

        self.ZTR = LTR * 2.0 - 1
        self.nFeatures = DTR.shape[0]
        self.nSamples = DTR.shape[1]

        self.Hh = None
        self.alpha = None
        self.trained = False
        self.balanced_C = None

