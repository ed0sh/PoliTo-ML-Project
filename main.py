import numpy

from Utils import Plots, Preproccessing, Util
from Classifiers.ClassifiersInterface import ClassifiersInterface
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
            except Exception as e:
                print("Error while reading the file!")
                print(e)
                return

        D, L = numpy.hstack(DList), numpy.array(labelsList, dtype=numpy.int32)
        return D, L


def evaluate_model(DTR: numpy.array, LTR: numpy.array, PCA_values: list, K: int, modelObject: ClassifiersInterface, scaled_workPoint: Util.WorkPoint):
    print("PCA\t|\tminDCF")
    for m in PCA_values:
        reduced_DTR = DTR
        if m is not None:
            reduced_DTR = Preproccessing.PCA(DTR, m)[0]
        else:
            m = "No"

        modelObject.update_dataset(reduced_DTR, LTR)
        error, DCF, minDCF = Util.k_folds(reduced_DTR, LTR, K, modelObject, scaled_workPoint)
        minDCF = round(minDCF, 3)
        print(f"{m}\t|\t{minDCF}")


if __name__ == '__main__':
    # --- Load the dataset ---
    (DTR, LTR) = readfile('data/Train.csv')
    (DTE, LTE) = readfile('data/Test.csv')

    Z_DTR = Preproccessing.Z_Score(DTR)
    Z_DTE = Preproccessing.Z_Score(DTE)

    # --- Dataset exploration ---

    # PCA
    Plots.pair_plot(DTR, LTR)
    reduced_DTR = Preproccessing.PCA(DTR, 2)[0]
    Plots.pair_plot(reduced_DTR, LTR)

    # LDA
    reduced_DTR_LDA = Preproccessing.LDA(DTR, LTR, 1)[0]
    Plots.pair_plot(Util.vrow(reduced_DTR_LDA), LTR)

    # Correlation matrices
    dataset_cov_matrix = Util.dataCorrelationMatrix(DTR)
    Plots.plot_correlation_matrix(dataset_cov_matrix, "Dataset")

    for label in numpy.unique(LTR):
        class_cov_matrix = Util.dataCorrelationMatrix(DTR[:, LTR == label])
        Plots.plot_correlation_matrix(class_cov_matrix, f"Class {label}")

    # Choosing the number of PCA features
    expl_variance_fract = []
    for i in range(DTR.shape[0] + 1):
        expl_variance_fract.append(Preproccessing.PCA(DTR, i)[1])

    Plots.plot_simple_plot(range(DTR.shape[0] + 1), expl_variance_fract, "PCA components", "Fraction of explained variance")
    # Acceptable values are 7, 8 and 9 as they retain at least 90% of the dataset variance
    PCA_values = [None, 9, 8, 7]

    # Set the application-specific working point
    workPoint = Util.WorkPoint(0.5, 1, 10)
    scaled_workPoint = Util.WorkPoint(workPoint.effective_prior(), 1, 1)
    K = 5

    print("----- Dataset Correlation -----")
    pairs, mean = Util.evaluateClassCorrelation(DTR, LTR, 0.5)
    print(f"Feature Pairs over threshold : {pairs}\nCorrelation Mean : {mean}")

    # Model evaluation

    print("----- MVG -----")
    logMVG = MVGClassifier(DTR, LTR, scaled_workPoint.pi)
    evaluate_model(DTR, LTR, PCA_values, K, logMVG, scaled_workPoint)

    # The best minDCF value was given by m = 7 => we try also 6 and 5
    evaluate_model(DTR, LTR, [6, 5], K, logMVG, scaled_workPoint)
    # No, we can discard them

    # FIXME: Il prof non ha messo lo z-score su MVG (@ale dimmi tu)
    # print("----- MVG With Z-Score -----")
    # logMVG = MVGClassifier(Z_DTR, LTR, scaled_workPoint.pi)
    # error, DCF, minDCF = Util.k_folds(Z_DTR, LTR, K, logMVG, scaled_workPoint)
    # print(f"Error rate : {error} \nNormalized DCF : {DCF}\nMinDCF : {minDCF}")

    print("----- Naive MVG -----")
    logNaiveMVG = NaiveMVGClassifier(DTR, LTR, scaled_workPoint.pi)
    evaluate_model(DTR, LTR, [10], K, logNaiveMVG, scaled_workPoint)
    evaluate_model(DTR, LTR, PCA_values, K, logNaiveMVG, scaled_workPoint)

    # FIXME: Il prof non ha messo lo z-score su MVG (@ale dimmi tu)
    # print("----- Naive MVG with ZScore -----")
    # logNaiveMVG = NaiveMVGClassifier(Z_DTR, LTR, scaled_workPoint.pi)
    # error, DCF, minDCF = Util.k_folds(Z_DTR, LTR, K, logNaiveMVG, scaled_workPoint)
    # print(f"Error rate : {error} \nNormalized DCF : {DCF}\nMinDCF : {minDCF}")

    print("----- TiedMVG -----")
    logTiedMVG = TiedMVGClassifier(DTR, LTR, scaled_workPoint.pi)
    evaluate_model(DTR, LTR, PCA_values, K, logTiedMVG, scaled_workPoint)

    # FIXME: Il prof non ha messo lo z-score su MVG (@ale dimmi tu)
    # print("----- TiedMVG with ZScore-----")
    # logTiedMVG = TiedMVGClassifier(Z_DTR, LTR, scaled_workPoint.pi)
    # error, DCF, minDCF = Util.k_folds(Z_DTR, LTR, K, logTiedMVG, scaled_workPoint)
    # print(f"Error rate : {error} \nNormalized DCF : {DCF}\nMinDCF : {minDCF}")

    print("----- log Regression -----")
    logReg = LogRegClass(DTR, LTR, 0.00001)
    evaluate_model(DTR, LTR, PCA_values, K, logReg, scaled_workPoint)

    print("----- log Regression with ZScore-----")
    logReg = LogRegClass(Z_DTR, LTR, 0.00001)
    evaluate_model(Z_DTR, LTR, PCA_values, K, logReg, scaled_workPoint)

    print("----- log Regression with optimized lambdas-----")
    logReg = LogRegClass.optimize_lambda(DTR, LTR, scaled_workPoint)
    evaluate_model(DTR, LTR, PCA_values, K, logReg, scaled_workPoint)

    print("----- log Regression with inplace optimized lambdas-----")
    logReg = LogRegClass(DTR, LTR, 0.00001)
    logReg.optimize_lambda_inplace(scaled_workPoint, starting_lambda=0.001, offset=10, num_iterations=10000,
                                   tolerance=1e-3)
    evaluate_model(DTR, LTR, PCA_values, K, logReg, scaled_workPoint)
