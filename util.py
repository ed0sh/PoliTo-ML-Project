import numpy
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from Classifiers import ClassifiersInterface


def vcol(v: numpy.array):
    return v.reshape((v.size, 1))


def vrow(v: numpy.array):
    return v.reshape((1, v.size))


class WorkPoint:
    def __init__(self, pi: float, C_fn: float, C_fp: float):
        self.pi = pi
        self.C_fn = C_fn
        self.C_fp = C_fp

    def effective_prior(self):
        return (self.pi * self.C_fn) / (self.pi * self.C_fn + (1 - self.pi) * self.C_fp)

def dataCovarianceMatrix(D: numpy.array):
    mu = (D.mean(1))
    DC = D - vcol(mu)
    C = numpy.dot(DC, DC.T) / DC.shape[1]
    return C, vcol(mu)


def dataCorrelationMatrix(D: numpy.array):
    C, _ = dataCovarianceMatrix(D)
    std = numpy.sqrt(np.diag(C))
    return C / np.outer(std, std)


def evaluateCorrelation(D: numpy.array, threshold: float):
    correlation_matrix = dataCorrelationMatrix(D)
    featurePairsOver = (numpy.abs(correlation_matrix - numpy.diag(numpy.ones(D.shape[0]))) > threshold).astype(
        int).sum() / 2
    meanCorrelation = correlation_matrix.sum() / correlation_matrix.shape[0] ** 2
    return featurePairsOver.astype(int), meanCorrelation


def evaluateClassCorrelation(D: numpy.array, L: numpy.array, threshold: float):
    meanCorrelations = []
    featurePairsOver = []
    for label in np.unique(L):
        pairs, mean = evaluateCorrelation(D[:, L == label], threshold)
        meanCorrelations.append(mean)
        featurePairsOver.append(pairs)
    return featurePairsOver, meanCorrelations


def within_class_covariance(D: numpy.array, N: int):
    return dataCovarianceMatrix(D)[0] * D.size / N


def PCA(D: numpy.array, m: int):
    C, _ = dataCovarianceMatrix(D)
    U, _, _ = numpy.linalg.svd(C)
    P = U[:, 0:m]
    return numpy.dot(P.T, D)


def Z_Score(D: numpy.array):
    for i in range(D.shape[0]):
        D[i] = (D[i] - D[i].mean()) / D[i].std()
    return D


def Compute_Anormalized_DCF(matrix: numpy.array, pi: float, C_fn: float, C_fp: float):
    FNR = matrix[0][1] / (matrix[0][1] + matrix[1][1])
    FPR = matrix[1][0] / (matrix[0][0] + matrix[1][0])

    DCF = pi * C_fn * FNR + (1 - pi) * C_fp * FPR
    return DCF


def Compute_Normalized_DCF(DCF: numpy.array, pi: float, C_fn: float, C_fp: float):
    optimal_risk = numpy.min([pi * C_fn, (1 - pi) * C_fp])
    return DCF / optimal_risk


def Compute_DCF(matrix: numpy.array, workPoint: WorkPoint):
    DCF = Compute_Anormalized_DCF(matrix, workPoint.pi, workPoint.C_fn, workPoint.C_fp)
    nDCF = Compute_Normalized_DCF(DCF, workPoint.pi, workPoint.C_fn, workPoint.C_fp)
    return DCF, nDCF


def logpdf_GAU_ND(X: numpy.array, mu: float, C: numpy.array):
    _, log_determinant = numpy.linalg.slogdet(C)
    firstTerm = - (numpy.shape(X)[0] * 0.5) * numpy.log(2 * numpy.pi) - log_determinant * 0.5
    L = numpy.linalg.inv(C)
    XC = X - mu
    return firstTerm - 0.5 * (XC * numpy.dot(L, XC)).sum(0)


def split_db_2to1(D: numpy.array, L: numpy.array, seed=0):
    nTrain = int(D.shape[1] * 2.0 / 3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]  # Data Training
    DTE = D[:, idxTest]  # Data Test
    LTR = L[idxTrain]  # Label Training
    LTE = L[idxTest]  # Label Test
    return (DTR, LTR), (DTE, LTE)


def confusion_matrix(LTE: numpy.array, SPost: numpy.array):
    n = numpy.unique(LTE).shape[0]
    matrix = numpy.zeros([n, n])
    for i in numpy.unique(LTE):
        for j in numpy.unique(LTE):
            matrix[i][j] = (SPost.flat[LTE == j] == i).sum()
    return matrix


def split_k_folds(DTR: numpy.array, LTR: numpy.array, K: int, seed=0):
    d_folds = []
    l_folds = []
    new_k = K
    for i in range(len(numpy.unique(LTR))):
        d_folds.append([])
        l_folds.append([])

        DTR_L = DTR[:, LTR == i]
        LTR_L = LTR[LTR == i]
        # if k is > than the number of elements, then cap it to the num of elements
        # in order to have one sample for each group
        new_k = min(K, DTR_L.shape[1])
        nTrain = int(DTR_L.shape[1] / new_k)
        numpy.random.seed(seed)
        idx = numpy.random.permutation(DTR_L.shape[1])

        # split the dataset into k parts of nTrain elements
        j = 0
        for j in range(new_k):
            idxTrain = idx[(j * nTrain): ((j + 1) * nTrain)]
            d_folds[i].append(DTR_L[:, idxTrain])
            l_folds[i].append(LTR_L[idxTrain])

        # if there are some elements that were left behind,
        # then distribute them equally
        if sum([d_folds[i][k].shape[1] for k in range(new_k)], 0) < DTR_L.shape[1]:
            j += 1
            idxTrain = idx[(j * nTrain):]
            d_remaining = DTR_L[:, idxTrain]
            l_remaining = LTR_L[idxTrain]
            for x in range(d_remaining.shape[1]):
                d_folds[i][x] = numpy.hstack([d_folds[i][x], vcol(d_remaining[:, x])])
                l_folds[i][x] = numpy.hstack([l_folds[i][x], l_remaining[x]])

    d_result = d_folds[0].copy()
    l_result = l_folds[0].copy()

    # if k was capped in the previous step and the first row of the dataset is not equal to K,
    # then create a set composed by a number of elements that's equal to the original K
    taken = 0
    if len(d_result) < K:
        for j in range(1, len(d_folds)):
            el = 0
            while len(d_result) < K and el < len(d_folds[j]):
                d_result.append(d_folds[j][el])
                l_result.append(l_folds[j][el])
                taken += 1
                el += 1

    # if the result set didn't take into account all the elements,
    # then distribute them equally
    if len(d_result) < DTR.shape[1]:
        index = 0
        group = int(taken / len(d_folds[1]))
        offset = int(taken % len(d_folds[1]))
        for i in range(1 + group, len(d_folds)):
            for j in range(offset if i == 1 else 0, len(d_folds[i])):
                d_result[index] = numpy.hstack([d_result[index], d_folds[i][j]])
                l_result[index] = numpy.hstack([l_result[index], l_folds[i][j]])
                index = (index + 1) % K

    return d_result, l_result


def k_folds(DTR: numpy.array, LTR: numpy.array, K: int, modelObject: ClassifiersInterface, workPoint: WorkPoint):
    d_folds, l_folds = split_k_folds(DTR, LTR, K)
    DCFs = []
    error_rates = []
    for i in range(len(d_folds)):
        data_test_set = d_folds[i]
        labels_test_set = l_folds[i]
        if i == 0:
            data_train_set = numpy.hstack(d_folds[i + 1:])
            labels_train_set = numpy.hstack(l_folds[i + 1:])
        elif i < len(d_folds) - 1:
            data_train_set = numpy.hstack([numpy.hstack(d_folds[0:i]), numpy.hstack(d_folds[i + 1:])])
            labels_train_set = numpy.hstack([numpy.hstack(l_folds[0:i]), numpy.hstack(l_folds[i + 1:])])
        else:
            data_train_set = numpy.hstack(d_folds[0:i])
            labels_train_set = numpy.hstack(l_folds[0:i])

        modelObject.update_dataset(data_train_set, labels_train_set)
        modelObject.train()
        predicted = modelObject.classify(data_test_set)
        err_rate, DCF = evaluate(predicted, labels_test_set, workPoint)
        error_rates.append(err_rate)
        DCFs.append(DCF)

    mean_err_rate = numpy.array(error_rates).mean()
    mean_DCF = numpy.array(DCFs).mean()
    return mean_err_rate, mean_DCF


def plot_roc_curve(FPRs: numpy.array, TPRs: numpy.array):
    plt.figure()
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.plot(FPRs, TPRs)
    plt.grid()
    plt.show()


def plot_bayes_error(effPriorLogOdds: numpy.array, DCFs: numpy.array, minDCFs: numpy.array):
    plt.figure()
    plt.plot(effPriorLogOdds, DCFs, label='DCF', color='r')
    plt.plot(effPriorLogOdds, minDCFs, label='min DCF', color='b')
    plt.legend()
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.show()


def plot_scatter(DTR: numpy.array, LTR: numpy.array):
    f0 = DTR[:, LTR == 0]
    f1 = DTR[:, LTR == 1]

    with PdfPages('target/scatters.pdf') as pdf:
        for i in range(DTR.shape[0]):
            for j in range(i, DTR.shape[0]):
                if i == j:
                    continue
                fig = plt.figure()
                plt.xlabel(f"Feature: {i}")
                plt.ylabel(f"Feature: {j}")
                plt.scatter(f0[i, :], f0[j, :], label='Spoofed')
                plt.scatter(f1[i, :], f1[j, :], label='Authentic')

                plt.legend()
                pdf.savefig(fig)
                plt.close(fig)


def plot_hists(DTR: numpy.array, LTR: numpy.array):
    f0 = DTR[:, LTR == 0]
    f1 = DTR[:, LTR == 1]

    with PdfPages('target/hists.pdf') as pdf:
        for i in range(DTR.shape[0]):
            fig = plt.figure()
            plt.xlabel(f"Feature: {i}")
            plt.ylabel("Density")
            plt.hist(f0[i, :], density=True, bins=20, label='Spoofed', alpha=0.4)
            plt.hist(f1[i, :], density=True, bins=20, label='Authentic', alpha=0.4)

            plt.legend()
            pdf.savefig(fig)
            plt.close(fig)


def evaluate(PLabel: numpy.array, LTE: numpy.array, workPoint: WorkPoint):
    errRate = ((LTE != PLabel).astype(int).sum() / LTE.shape[0]) * 100
    matrix = confusion_matrix(LTE, PLabel)
    _, DCF = Compute_DCF(matrix, workPoint)
    return errRate, DCF
