import numpy

from Utils import Plots, Preproccessing, Util
from Classifiers.LogisticRegression import LogRegClass
from Classifiers.MVGClassifier import MVGClassifier
from Classifiers.NaiveMVGClassifier import NaiveMVGClassifier
from Classifiers.TiedMVGClassifier import TiedMVGClassifier


def readfile(file):
    DList = []
    labelsList = []
    hLabels = {
        '0': 0,
        '1': 1
    }

    with open(file) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:10]
                attrs = Util.vcol(numpy.array([float(i) for i in attrs]))
                name = line.split(',')[-1].strip()
                label = hLabels[name]
                DList.append(attrs)
                labelsList.append(label)
            except:
                pass
        D, L = numpy.hstack(DList), numpy.array(labelsList, dtype=numpy.int32)
        return D, L


if __name__ == '__main__':
    # --- Load the dataset ---
    (DTR, LTR) = readfile('data/Train.csv')
    (DTE, LTE) = readfile('data/Test.csv')

    Z_DTR = Preproccessing.Z_Score(DTR)
    Z_DTE = Preproccessing.Z_Score(DTE)

    # --- Dataset exploration ---
    Plots.pair_plot(DTR, LTR)
    reduced_DTR = Preproccessing.PCA(DTR, 2)[0]
    Plots.pair_plot(reduced_DTR, LTR)
    # TODO: plot density after LDA

    # Correlation matrices
    dataset_cov_matrix = Util.dataCorrelationMatrix(DTR)
    Plots.plot_correlation_matrix(dataset_cov_matrix, "Dataset")

    for label in numpy.unique(LTR):
        class_cov_matrix = Util.dataCorrelationMatrix(DTR[:, LTR == label])
        Plots.plot_correlation_matrix(class_cov_matrix, f"Class {label}")

    expl_variance_fract = []
    for i in range(DTR.shape[0] + 1):
        expl_variance_fract.append(Preproccessing.PCA(DTR, i)[1])

    Plots.plot_simple_plot(range(DTR.shape[0] + 1), expl_variance_fract, "PCA components", "Fraction of explained variance")

    workPoint = Util.WorkPoint(0.5, 1, 10)
    scaled_workPoint = Util.WorkPoint(workPoint.effective_prior(), 1, 1)
    K = 5

    print("----- Dataset Correlation -----")
    pairs, mean = Util.evaluateClassCorrelation(DTR, LTR, 0.5)
    print(f"Feature Pairs over threshold : {pairs}\nCorrelation Mean : {mean}")

    print("----- MVG -----")
    logMVG = MVGClassifier(DTR, LTR, scaled_workPoint.pi)
    error, DCF = Util.k_folds(DTR, LTR, K, logMVG, scaled_workPoint)
    print(f"Error rate : {error} \nNormalized DCF : {DCF}")

    print("----- MVG With Z-Score -----")
    logMVG = MVGClassifier(Z_DTR, LTR, scaled_workPoint.pi)
    error, DCF = Util.k_folds(Z_DTR, LTR, K, logMVG, scaled_workPoint)
    print(f"Error rate : {error} \nNormalized DCF : {DCF}")

    print("----- Naive MVG -----")
    logNaiveMVG = NaiveMVGClassifier(DTR, LTR, scaled_workPoint.pi)
    error, DCF = Util.k_folds(DTR, LTR, K, logNaiveMVG, scaled_workPoint)
    print(f"Error rate : {error} \nNormalized DCF : {DCF}")

    print("----- Naive MVG with ZScore -----")
    logNaiveMVG = NaiveMVGClassifier(Z_DTR, LTR, scaled_workPoint.pi)
    error, DCF = Util.k_folds(Z_DTR, LTR, K, logNaiveMVG, scaled_workPoint)
    print(f"Error rate : {error} \nNormalized DCF : {DCF}")

    print("----- TiedMVG -----")
    logTiedMVG = TiedMVGClassifier(DTR, LTR, scaled_workPoint.pi)
    error, DCF = Util.k_folds(DTR, LTR, K, logTiedMVG, scaled_workPoint)
    print(f"Error rate : {error} \nNormalized DCF : {DCF}")

    print("----- TiedMVG with ZScore-----")
    logTiedMVG = TiedMVGClassifier(Z_DTR, LTR, scaled_workPoint.pi)
    error, DCF = Util.k_folds(Z_DTR, LTR, K, logTiedMVG, scaled_workPoint)
    print(f"Error rate : {error} \nNormalized DCF : {DCF}")

    print("----- log Regression -----")
    logReg = LogRegClass(DTR, LTR, 0.00001)
    error, DCF = Util.k_folds(DTR, LTR, K, logReg, scaled_workPoint)
    print(f"Error rate : {error} \nNormalized DCF : {DCF}")

    print("----- log Regression with ZScore-----")
    logReg = LogRegClass(Z_DTR, LTR, 0.00001)
    error, DCF = Util.k_folds(Z_DTR, LTR, K, logReg, scaled_workPoint)
    print(f"Error rate : {error} \nNormalized DCF : {DCF}")

    print("----- log Regression with optimized lambdas-----")
    logReg = LogRegClass.optimize_lambda(DTR, LTR, scaled_workPoint)
    error, DCF = Util.k_folds(DTR, LTR, K, logReg, scaled_workPoint)
    print(f"Error rate : {error} \nNormalized DCF : {DCF} \nLambda : {logReg.lam}")

    print("----- log Regression with K-Folds optimized lambdas-----")
    logReg = LogRegClass(DTR, LTR, 0.00001)
    logReg.optimize_lambda_inplace(scaled_workPoint, starting_lambda=0.001, offset=10, num_iterations=10000,
                                   tolerance=1e-3)
    error, DCF = Util.k_folds(DTR, LTR, K, logReg, scaled_workPoint)
    print(f"Error rate : {error} \nNormalized DCF : {DCF} \nLambda : {logReg.lam}")
