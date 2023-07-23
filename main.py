import numpy

from Utils import Plots, Preproccessing, Util
from Classifiers.ClassifiersInterface import ClassifiersInterface
from Classifiers.LogisticRegression import LogRegClass
from Classifiers.MVGClassifier import MVGClassifier
from Classifiers.NaiveMVGClassifier import NaiveMVGClassifier
from Classifiers.TiedMVGClassifier import TiedMVGClassifier

if __name__ == '__main__':
    # --- Load the dataset ---
    (DTR, LTR) = Util.readfile('data/Train.csv')
    (DTE, LTE) = Util.readfile('data/Test.csv')

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

    Plots.plot_simple_plot(
        range(DTR.shape[0] + 1),
        expl_variance_fract,
        x_label="PCA components",
        y_label="Fraction of explained variance",
        color='green',
        title=f"PCA vs. Fraction of explained variance",
        x_scale="linear"
    )

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
    Util.evaluate_model(DTR, LTR, PCA_values, K, logMVG, scaled_workPoint)

    # The best minDCF value was given by m = 7 => we try also 6 and 5
    Util.evaluate_model(DTR, LTR, [6, 5], K, logMVG, scaled_workPoint)
    # No, we can discard them

    # FIXME: Il prof non ha messo lo z-score su MVG (@ale dimmi tu)
    # print("----- MVG With Z-Score -----")
    # logMVG = MVGClassifier(Z_DTR, LTR, scaled_workPoint.pi)
    # error, DCF, minDCF = Util.k_folds(Z_DTR, LTR, K, logMVG, scaled_workPoint)
    # print(f"Error rate : {error} \nNormalized DCF : {DCF}\nMinDCF : {minDCF}")

    print("----- Naive MVG -----")
    logNaiveMVG = NaiveMVGClassifier(DTR, LTR, scaled_workPoint.pi)
    Util.evaluate_model(DTR, LTR, [10], K, logNaiveMVG, scaled_workPoint)
    Util.evaluate_model(DTR, LTR, PCA_values, K, logNaiveMVG, scaled_workPoint)

    # FIXME: Il prof non ha messo lo z-score su MVG (@ale dimmi tu)
    # print("----- Naive MVG with ZScore -----")
    # logNaiveMVG = NaiveMVGClassifier(Z_DTR, LTR, scaled_workPoint.pi)
    # error, DCF, minDCF = Util.k_folds(Z_DTR, LTR, K, logNaiveMVG, scaled_workPoint)
    # print(f"Error rate : {error} \nNormalized DCF : {DCF}\nMinDCF : {minDCF}")

    print("----- TiedMVG -----")
    logTiedMVG = TiedMVGClassifier(DTR, LTR, scaled_workPoint.pi)
    Util.evaluate_model(DTR, LTR, PCA_values, K, logTiedMVG, scaled_workPoint)

    # FIXME: Il prof non ha messo lo z-score su MVG (@ale dimmi tu)
    # print("----- TiedMVG with ZScore-----")
    # logTiedMVG = TiedMVGClassifier(Z_DTR, LTR, scaled_workPoint.pi)
    # error, DCF, minDCF = Util.k_folds(Z_DTR, LTR, K, logTiedMVG, scaled_workPoint)
    # print(f"Error rate : {error} \nNormalized DCF : {DCF}\nMinDCF : {minDCF}")

    print("----- log Regression with optimized lambdas-----")
    logReg = LogRegClass.create_with_optimized_lambda(DTR, LTR, scaled_workPoint)
    Util.evaluate_model(DTR, LTR, PCA_values, K, logReg, scaled_workPoint)
    print(f"Selected lambda: {logReg.lam}")

    print("----- log Regression with ZScore and inplace optimized lambdas-----")
    logReg = LogRegClass.create_with_optimized_lambda(Z_DTR, LTR, scaled_workPoint)
    Util.evaluate_model(Z_DTR, LTR, PCA_values, K, logReg, scaled_workPoint)
    print(f"Selected lambda: {logReg.lam}")

    print("----- Quadratic log Regression with optimized lambdas-----")
    logReg = LogRegClass.create_with_optimized_lambda(DTR, LTR, scaled_workPoint)
    logReg.quadratic = True
    Util.evaluate_model(logReg.DTR, LTR, PCA_values, K, logReg, scaled_workPoint)
    print(f"Selected lambda: {logReg.lam}")

    print("----- Quadratic log Regression with ZScore and inplace optimized lambdas-----")
    logReg = LogRegClass.create_with_optimized_lambda(Z_DTR, LTR, scaled_workPoint)
    logReg.quadratic = True
    Util.evaluate_model(logReg.DTR, LTR, PCA_values, K, logReg, scaled_workPoint)
    print(f"Selected lambda: {logReg.lam}")

    # SVMs
    C_vec = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2]
    K_svm_vec = [1, 1e1]
    colors = ['red', 'orange', 'blue', 'green']
    PCA_values_reduced = [None, 7]

    print("----- SVM linear -----")
    Util.svm_cross_val_graphs(K_svm_vec, C_vec, DTR, LTR, PCA_values_reduced, scaled_workPoint, False,
                              colors, svm_type_label="SVM linear")

    print("----- SVM linear re-balanced -----")
    Util.svm_cross_val_graphs(K_svm_vec, C_vec, DTR, LTR, PCA_values_reduced, scaled_workPoint, True,
                              colors, svm_type_label="SVM linear re-balanced")

    print("----- SVM linear w/ Z-score -----")
    Util.svm_cross_val_graphs(K_svm_vec, C_vec, Z_DTR, LTR, PCA_values_reduced, scaled_workPoint, False,
                              colors, svm_type_label="SVM linear w/ Z-score")

    print("----- SVM linear re-balanced w/ Z-score -----")
    Util.svm_cross_val_graphs(K_svm_vec, C_vec, Z_DTR, LTR, PCA_values_reduced, scaled_workPoint, True,
                              colors, svm_type_label="SVM linear re-balanced w/ Z-score")

    C_vec = [1e-2, 1e-1, 1, 1e1]
    c = 0
    print("----- SVM poly 2 -----")
    Util.svm_cross_val_graphs(K_svm_vec, C_vec, DTR, LTR, PCA_values_reduced, scaled_workPoint, False,
                              colors, svm_type_label="SVM poly 2", kernel_type="poly", d=2, c=c)

    print("----- SVM poly 2 re-balanced -----")
    Util.svm_cross_val_graphs(K_svm_vec, C_vec, DTR, LTR, PCA_values_reduced, scaled_workPoint, True,
                              colors, svm_type_label="SVM poly 2 re-balanced", kernel_type="poly", d=2, c=c)

    print("----- SVM poly 2 w/ Z-score -----")
    Util.svm_cross_val_graphs(K_svm_vec, C_vec, Z_DTR, LTR, PCA_values_reduced, scaled_workPoint, False,
                              colors, svm_type_label="SVM poly 2 w/ Z-score", kernel_type="poly", d=2, c=c)

    print("----- SVM poly 2 re-balanced w/ Z-score -----")
    Util.svm_cross_val_graphs(K_svm_vec, C_vec, Z_DTR, LTR, PCA_values_reduced, scaled_workPoint, True,
                              colors, svm_type_label="SVM poly 2 re-balanced w/ Z-score", kernel_type="poly", d=2, c=c)

    C_vec = [1e-3, 1e-2, 1e-1, 1]
    c_vec = [1, 10]
    for c in c_vec:
        print(f"----- SVM poly 2 - c={c} -----")
        Util.svm_cross_val_graphs(K_svm_vec, C_vec, DTR, LTR, PCA_values_reduced, scaled_workPoint, False,
                                  colors, svm_type_label=f"SVM poly 2 - c={c}", kernel_type="poly", d=2, c=c)

        print(f"----- SVM poly 2 re-balanced - c={c} -----")
        Util.svm_cross_val_graphs(K_svm_vec, C_vec, DTR, LTR, PCA_values_reduced, scaled_workPoint, True,
                                  colors, svm_type_label=f"SVM poly 2 re-balanced - c={c}", kernel_type="poly", d=2, c=c)

    C_vec = [1e-5, 1e-4, 1e-3, 1e-2]
    c_vec = [1, 10]
    for c in c_vec:
        print(f"----- SVM poly 3 - c={c} -----")
        Util.svm_cross_val_graphs(K_svm_vec, C_vec, DTR, LTR, PCA_values_reduced, scaled_workPoint, False,
                                  colors, svm_type_label=f"SVM poly 3 - c={c}", kernel_type="poly", d=3, c=c)

        print(f"----- SVM poly 3 re-balanced - c={c} -----")
        Util.svm_cross_val_graphs(K_svm_vec, C_vec, Z_DTR, LTR, PCA_values_reduced, scaled_workPoint, False,
                                  colors, svm_type_label=f"SVM poly 3 w/ Z-score - c={c}", kernel_type="poly", d=3,
                                  c=c)

    C_vec = [1e-4, 1e-3, 2e-3, 1e-2, 2e-2, 1e-1, 2e-1, 1, 2]
    K_svm_vec = [1e-3, 1e-2, 0.05, 0.1]
    colors = ['red', 'orange', 'blue', 'green', 'black', 'yellow', 'grey', 'purple']

    log_gamma_values = [-3, -4, -5, -6]
    for log_gamma in log_gamma_values:
        print(f"----- SVM RBF - log ɣ={log_gamma} -----")
        Util.svm_cross_val_graphs(K_svm_vec, C_vec, DTR, LTR, PCA_values_reduced, scaled_workPoint, False,
                                  colors, svm_type_label=f"SVM RBF - log ɣ={log_gamma}", kernel_type="rbf",
                                  gamma=numpy.exp(log_gamma))

