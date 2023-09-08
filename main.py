import numpy

from Utils import Plots, Preproccessing, Util
from Classifiers.ClassifiersInterface import ClassifiersInterface
from Classifiers.LogisticRegression import LogRegClass
from Classifiers.MVGClassifier import MVGClassifier
from Classifiers.NaiveMVGClassifier import NaiveMVGClassifier
from Classifiers.TiedMVGClassifier import TiedMVGClassifier
from Classifiers.LinearSVM import LinearSVM
from Classifiers.KernelSVM import KernelSVM
from Classifiers.GMMClassifier import GMMClassifier

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
        x_scale="linear",
        y_scale="linear"
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

    # --- Model evaluation ---

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
    Util.svm_cross_val_graphs(K_svm_vec, C_vec, DTR, LTR, PCA_values_reduced, K, scaled_workPoint, False,
                              colors, svm_type_label="SVM linear")

    print("----- SVM linear re-balanced -----")
    Util.svm_cross_val_graphs(K_svm_vec, C_vec, DTR, LTR, PCA_values_reduced, K, scaled_workPoint, True,
                              colors, svm_type_label="SVM linear re-balanced")

    print("----- SVM linear w/ Z-score -----")
    Util.svm_cross_val_graphs(K_svm_vec, C_vec, Z_DTR, LTR, PCA_values_reduced, K, scaled_workPoint, False,
                              colors, svm_type_label="SVM linear w/ Z-score")

    print("----- SVM linear re-balanced w/ Z-score -----")
    Util.svm_cross_val_graphs(K_svm_vec, C_vec, Z_DTR, LTR, PCA_values_reduced, K, scaled_workPoint, True,
                              colors, svm_type_label="SVM linear re-balanced w/ Z-score")

    C_vec = [1e-2, 1e-1, 1, 1e1]
    c = 0
    print("----- SVM poly 2 -----")
    Util.svm_cross_val_graphs(K_svm_vec, C_vec, DTR, LTR, PCA_values_reduced, K, scaled_workPoint, False,
                              colors, svm_type_label="SVM poly 2", kernel_type="poly", d=2, c=c)

    print("----- SVM poly 2 re-balanced -----")
    Util.svm_cross_val_graphs(K_svm_vec, C_vec, DTR, LTR, PCA_values_reduced, K, scaled_workPoint, True,
                              colors, svm_type_label="SVM poly 2 re-balanced", kernel_type="poly", d=2, c=c)

    print("----- SVM poly 2 w/ Z-score -----")
    Util.svm_cross_val_graphs(K_svm_vec, C_vec, Z_DTR, LTR, PCA_values_reduced, K, scaled_workPoint, False,
                              colors, svm_type_label="SVM poly 2 w/ Z-score", kernel_type="poly", d=2, c=c)

    print("----- SVM poly 2 re-balanced w/ Z-score -----")
    Util.svm_cross_val_graphs(K_svm_vec, C_vec, Z_DTR, LTR, PCA_values_reduced, K, scaled_workPoint, True,
                              colors, svm_type_label="SVM poly 2 re-balanced w/ Z-score", kernel_type="poly", d=2, c=c)

    C_vec = [1e-3, 1e-2, 1e-1, 1]
    c_vec = [1, 10]
    for c in c_vec:
        print(f"----- SVM poly 2 - c={c} -----")
        Util.svm_cross_val_graphs(K_svm_vec, C_vec, DTR, LTR, PCA_values_reduced, K, scaled_workPoint, False,
                                  colors, svm_type_label=f"SVM poly 2 - c={c}", kernel_type="poly", d=2, c=c)

        print(f"----- SVM poly 2 re-balanced - c={c} -----")
        Util.svm_cross_val_graphs(K_svm_vec, C_vec, DTR, LTR, PCA_values_reduced, K, scaled_workPoint, True,
                                  colors, svm_type_label=f"SVM poly 2 re-balanced - c={c}", kernel_type="poly", d=2,
                                  c=c)

    C_vec = [1e-5, 1e-4, 1e-3, 1e-2]
    c_vec = [1, 10]
    for c in c_vec:
        print(f"----- SVM poly 3 - c={c} -----")
        Util.svm_cross_val_graphs(K_svm_vec, C_vec, DTR, LTR, PCA_values_reduced, K, scaled_workPoint, False,
                                  colors, svm_type_label=f"SVM poly 3 - c={c}", kernel_type="poly", d=3, c=c)

        print(f"----- SVM poly 3 re-balanced - c={c} -----")
        Util.svm_cross_val_graphs(K_svm_vec, C_vec, Z_DTR, LTR, PCA_values_reduced, K, scaled_workPoint, False,
                                  colors, svm_type_label=f"SVM poly 3 w/ Z-score - c={c}", kernel_type="poly", d=3,
                                  c=c)

    C_vec = [1e-4, 1e-3, 2e-3, 1e-2, 2e-2, 1e-1, 2e-1, 1, 2]
    K_svm_vec = [1e-3, 1e-2, 0.05, 0.1]
    colors = ['red', 'orange', 'blue', 'green', 'black', 'yellow', 'grey', 'purple']

    log_gamma_values = [-3, -4, -5, -6]
    for log_gamma in log_gamma_values:
        print(f"----- SVM RBF - log ɣ={log_gamma} -----")
        Util.svm_cross_val_graphs(K_svm_vec, C_vec, DTR, LTR, PCA_values_reduced, K, scaled_workPoint, False,
                                  colors, svm_type_label=f"SVM RBF - log ɣ={log_gamma}", kernel_type="rbf",
                                  gamma=numpy.exp(log_gamma))

    # GMMs
    params_gmm_target = {
        "alpha": 1e-1,
        "psi": 1e-2,
        "max_g": 1,
        "diagonal": False,
        "tied": False
    }

    params_gmm_non_target = {
        "alpha": 1e-1,
        "psi": 1e-2,
        "max_g": 1,
        "diagonal": False,
        "tied": False
    }

    max_g_c0_vec = [1, 2, 4, 8, 16, 32]
    # For visualization purposes, split max_g_c1_vec in 2 and run with max_g_c1_vec containing only 3 values at a time
    max_g_c1_vec = [1, 2, 4, 8, 16, 32]
    PCA_values = [None]

    print(f"----- GMM - target 'g' components: {max_g_c1_vec} -----")
    max_g_minDCFs = Util.gmm_grid_search_one_prop(DTR, LTR,
                                                  max_g_c0_vec, max_g_c1_vec,
                                                  "max_g",
                                                  params_gmm_target,
                                                  params_gmm_non_target,
                                                  PCA_values, K,
                                                  scaled_workPoint)
    sigma_type_t = Util.get_sigma_type_as_string(params_gmm_target["diagonal"], params_gmm_target["tied"])
    sigma_type_nt = Util.get_sigma_type_as_string(params_gmm_non_target["diagonal"], params_gmm_non_target["tied"])
    Plots.plot_gmm_g_grid_search(max_g_minDCFs, max_g_c1_vec, 'Target class "g" components',
                                 "max_g", f"Non-target: {sigma_type_nt} - Target: {sigma_type_t}",
                                 PCA_values[0])

    # Best number of components "max_g"
    params_gmm_target = {
        "alpha": 1e-1,
        "psi": 1e-2,
        "max_g": 1,
        "diagonal": False,
        "tied": False
    }

    params_gmm_non_target = {
        "alpha": 1e-1,
        "psi": 1e-2,
        "max_g": 4,
        "diagonal": False,
        "tied": False
    }

    # Evaluate dimensionality reduction
    print("----- GMM classifier -----")
    PCA_values = [None, 10, 9, 8, 7]
    gmmClassifier = GMMClassifier(DTR, LTR, params_gmm_target, params_gmm_non_target)
    Util.evaluate_model(gmmClassifier.DTR, LTR, PCA_values, K, gmmClassifier, scaled_workPoint)

    # Find the best alpha for each GMM
    print(f"----- GMM - alpha grid search -----")
    params_gmm_target = {
        "alpha": 1e-1,
        "psi": 1e-2,
        "max_g": 16,
        "diagonal": True,
        "tied": True
    }

    params_gmm_non_target = {
        "alpha": 1e-1,
        "psi": 1e-2,
        "max_g": 8,
        "diagonal": True,
        "tied": True
    }
    PCA_values = [7]
    sigma_type_t = Util.get_sigma_type_as_string(params_gmm_target["diagonal"], params_gmm_target["tied"])
    sigma_type_nt = Util.get_sigma_type_as_string(params_gmm_non_target["diagonal"], params_gmm_non_target["tied"])

    alpha_vec_c0 = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1]
    # For visualization purposes, split alpha_vec_1 in 4 and run with alpha_vec_c1 containing only 2 values at a time
    alpha_vec_c1 = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1]
    alpha_minDCFs = Util.gmm_grid_search_one_prop(DTR, LTR,
                                                  alpha_vec_c0, alpha_vec_c1,
                                                  "alpha",
                                                  params_gmm_target,
                                                  params_gmm_non_target,
                                                  PCA_values, K,
                                                  scaled_workPoint)
    Plots.plot_gmm_g_grid_search(alpha_minDCFs, alpha_vec_c1, 'Target alpha',
                                 "alpha", f"Non-target: {sigma_type_nt} - Target: {sigma_type_t}",
                                 PCA_values[0])

    # Best values for alpha are the ones in the previous parameters definition

    # --- Calibration evaluation ---

    # Compare min DCFs with actual DCFs using training set and k-folds
    PCA_values = [7]
    gmmClassifier = GMMClassifier(DTR, LTR, params_gmm_target, params_gmm_non_target)
    gmm_minDFCs, gmm_DCFs = Util.evaluate_model(gmmClassifier.DTR, LTR, PCA_values, K, gmmClassifier, scaled_workPoint)

    polySVM = KernelSVM(DTR, LTR, d=2, c=10, C=1e-2, K=10, kernel_type="poly")
    polySVM_minDFCs, polySVM_DCFs = Util.evaluate_model(polySVM.DTR, LTR, PCA_values, K, polySVM, scaled_workPoint)

    rbfSVM = KernelSVM(DTR, LTR, gamma=numpy.exp(-5), K=0.01, C=0.1, kernel_type="rbf")
    rbfSVM_minDFCs, rbfSVM_DCFs = Util.evaluate_model(rbfSVM.DTR, LTR, PCA_values, K, rbfSVM, scaled_workPoint)

    print("PCA\t|\tminDCF\t|\tactDCF\t|\tModel")
    print(f"7\t|\t{gmm_minDFCs[0]}\t|\t{gmm_DCFs[0]}\t|\tGMM")
    print(f"7\t|\t{polySVM_minDFCs[0]}\t|\t{polySVM_DCFs[0]}\t|\tPoly-2-SVM")
    print(f"7\t|\t{rbfSVM_minDFCs[0]}\t|\t{rbfSVM_DCFs[0]}\t|\tRBF-SVM")

    # Plot an overall Bayes error plot
    PCA_value = 7
    Plots.new_figure()
    gmmClassifier = GMMClassifier(DTR, LTR, params_gmm_target, params_gmm_non_target)
    Util.bayes_error_calibration_evaluation_k_folds(gmmClassifier.DTR, LTR, PCA_value, K, gmmClassifier,
                                                    scaled_workPoint, "blue")

    polySVM = KernelSVM(DTR, LTR, d=2, c=10, C=1e-2, K=10, kernel_type="poly")
    Util.bayes_error_calibration_evaluation_k_folds(polySVM.DTR, LTR, PCA_value, K, polySVM, scaled_workPoint, "red")

    rbfSVM = KernelSVM(DTR, LTR, gamma=numpy.exp(-5), K=0.01, C=0.1, kernel_type="rbf")
    Util.bayes_error_calibration_evaluation_k_folds(rbfSVM.DTR, LTR, PCA_value, K, rbfSVM, scaled_workPoint, "green")
    Plots.show_plot()

    # Plot DET graph
    fig = Plots.new_figure()
    Util.DET_plot(DTR, LTR, PCA_value, K, gmmClassifier, scaled_workPoint, fig, "blue")
    Util.DET_plot(DTR, LTR, PCA_value, K, polySVM, scaled_workPoint, fig, "red")
    Util.DET_plot(DTR, LTR, PCA_value, K, rbfSVM, scaled_workPoint, fig, "green")
    Plots.show_plot()

    # Get calibration models and plot calibrated Bayes error graph
    Plots.new_figure()
    _, kf_calibrated_scores_gmm, kf_shuffled_labels_gmm = Util.score_calibration_k_folds(
        DTR, LTR, PCA_value, K, gmmClassifier, scaled_workPoint, "blue")

    _, kf_calibrated_scores_polySVM, kf_shuffled_labels_polySVM = Util.score_calibration_k_folds(
        DTR, LTR, PCA_value, K, polySVM, scaled_workPoint, "red")

    _, kf_calibrated_scores_rbfSVM, kf_shuffled_labels_rbfSVM = Util.score_calibration_k_folds(
        DTR, LTR, PCA_value, K, rbfSVM, scaled_workPoint, "green")
    Plots.show_plot()

    t = - numpy.log(scaled_workPoint.pi / (1 - scaled_workPoint.pi))

    # GMM calibrated DCF and minDCF
    predicted = (kf_calibrated_scores_gmm > t).astype(int)
    _, DCF = Util.evaluate(predicted, kf_shuffled_labels_gmm, workPoint)
    print(f"PCA - KF calibrated scores - GMM DCF: {DCF}")
    print(f"PCA - KF calibrated scores - GMM minDCF: {Util.compute_minDCF(kf_shuffled_labels_gmm, kf_calibrated_scores_gmm, workPoint)[0]}")

    # Poly SVM calibrated DCF and minDCF
    predicted = (kf_calibrated_scores_polySVM > t).astype(int)
    _, DCF = Util.evaluate(predicted, kf_shuffled_labels_polySVM, workPoint)
    print(f"PCA - KF calibrated scores - Poly SVM DCF: {DCF}")
    print(f"PCA - KF calibrated scores - Poly SVM minDCF: {Util.compute_minDCF(kf_shuffled_labels_polySVM, kf_calibrated_scores_polySVM, workPoint)[0]}")

    # RBF SVM calibrated DCF and minDCF
    predicted = (kf_calibrated_scores_rbfSVM > t).astype(int)
    _, DCF = Util.evaluate(predicted, kf_shuffled_labels_rbfSVM, workPoint)
    print(f"PCA - KF calibrated scores - RBF SVM DCF: {DCF}")
    print(f"PCA - KF calibrated scores - RBF SVM minDCF: {Util.compute_minDCF(kf_shuffled_labels_rbfSVM, kf_calibrated_scores_rbfSVM, workPoint)[0]}")

    # --- Fusion ---

    gmmClassifier = GMMClassifier(DTR, LTR, params_gmm_target, params_gmm_non_target)
    polySVM = KernelSVM(DTR, LTR, d=2, c=10, C=1e-2, K=10, kernel_type="poly")
    rbfSVM = KernelSVM(DTR, LTR, gamma=numpy.exp(-5), K=0.01, C=0.1, kernel_type="rbf")

    print("Fusion: GMM - Poly")
    Util.evaluate_model_fusion(DTR, LTR, PCA_value, K, gmmClassifier, polySVM, None, scaled_workPoint)

    print("Fusion: GMM - RBF")
    Util.evaluate_model_fusion(DTR, LTR, PCA_value, K, gmmClassifier, rbfSVM, None, scaled_workPoint)

    print("Fusion: RBF - Poly")
    Util.evaluate_model_fusion(DTR, LTR, PCA_value, K, rbfSVM, polySVM, None, scaled_workPoint)

    print("Fusion: GMM - Poly - RBF")
    Util.evaluate_model_fusion(DTR, LTR, PCA_value, K, gmmClassifier, polySVM, rbfSVM, scaled_workPoint)

    # Bayes error plot comparing the fusion model with the ones we chose
    Plots.new_figure()
    _, _, _ = Util.score_calibration_k_folds(DTR, LTR, PCA_value, K, gmmClassifier, scaled_workPoint, "blue")
    _, _, _ = Util.score_calibration_k_folds(DTR, LTR, PCA_value, K, polySVM, scaled_workPoint, "red")
    _, _, _ = Util.score_calibration_k_folds(DTR, LTR, PCA_value, K, rbfSVM, scaled_workPoint, "green")
    Util.evaluate_model_fusion(DTR, LTR, PCA_value, K, gmmClassifier, polySVM, None, scaled_workPoint)

    # Vertical line indicating the working point
    Plots.plot_vertical_line(numpy.log(scaled_workPoint.pi / (1 - scaled_workPoint.pi)))
    Plots.show_plot()

    # Plot DET graph with the fusion
    fig = Plots.new_figure()
    Util.DET_plot(DTR, LTR, PCA_value, K, gmmClassifier, scaled_workPoint, fig, "blue")
    Util.DET_plot(DTR, LTR, PCA_value, K, polySVM, scaled_workPoint, fig, "red")
    Util.DET_plot(DTR, LTR, PCA_value, K, rbfSVM, scaled_workPoint, fig, "green")
    Util.DET_plot_fusion(DTR, LTR, PCA_value, K, gmmClassifier, polySVM, None, scaled_workPoint, fig, "orange")
    Plots.show_plot()

    # --- Final evaluation ---

    PCAed_DTR, _, P = Preproccessing.PCA(DTR, PCA_value)
    PCAed_DTE = numpy.dot(P.T, DTE)

    # GMM
    gmmClassifier = GMMClassifier(PCAed_DTR, LTR, params_gmm_target, params_gmm_non_target)
    gmmClassifier.train()
    gmmClassifier.classify(PCAed_DTE, scaled_workPoint)
    gmm_scores = Util.vrow(gmmClassifier.get_scores())

    gmm_calibration_model, _, _ = Util.score_calibration(gmm_scores, LTE, K, None, scaled_workPoint, "blue")
    gmm_calibration_model.classify(gmm_scores, scaled_workPoint)
    calibrated_scores_gmm = gmm_calibration_model.get_scores()

    predicted = (calibrated_scores_gmm > t).astype(int)
    err_rate, DCF = Util.evaluate(predicted, LTE, workPoint)
    minDCF, gmm_FNRs, gmm_FPRs = Util.compute_minDCF(LTE, calibrated_scores_gmm, workPoint)
    print(f"Calibrated - PCA - GMM DCF: {DCF}")
    print(f"Calibrated - PCA - GMM minDCF: {minDCF}")

    # Poly
    polySVM = KernelSVM(PCAed_DTR, LTR, d=2, c=10, C=1e-2, K=10, kernel_type="poly")
    polySVM.train()
    polySVM.classify(PCAed_DTE, scaled_workPoint)
    polySVM_scores = Util.vrow(polySVM.get_scores())

    polySVM_calibration_model, _, _ = Util.score_calibration(polySVM_scores, LTE, K, None, scaled_workPoint, "blue")
    polySVM_calibration_model.classify(polySVM_scores, scaled_workPoint)
    calibrated_scores_polySVM = polySVM_calibration_model.get_scores()

    predicted = (calibrated_scores_polySVM > t).astype(int)
    _, DCF = Util.evaluate(predicted, LTE, workPoint)
    minDCF, polySVM_FNRs, polySVM_FPRs = Util.compute_minDCF(LTE, calibrated_scores_polySVM, workPoint)
    print(f"Calibrated - PCA - polySVM DCF: {DCF}")
    print(f"Calibrated - PCA - polySVM minDCF: {minDCF}")

    # RBF
    rbfSVM = KernelSVM(PCAed_DTR, LTR, gamma=numpy.exp(-5), K=0.01, C=0.1, kernel_type="rbf")
    rbfSVM.train()
    rbfSVM.classify(PCAed_DTE, scaled_workPoint)
    rbfSVM_scores = Util.vrow(rbfSVM.get_scores())

    rbfSVM_calibration_model, _, _ = Util.score_calibration(rbfSVM_scores, LTE, K, None, scaled_workPoint, "blue")
    rbfSVM_calibration_model.classify(rbfSVM_scores, scaled_workPoint)
    calibrated_scores_rbfSVM = rbfSVM_calibration_model.get_scores()

    predicted = (calibrated_scores_rbfSVM > t).astype(int)
    _, DCF = Util.evaluate(predicted, LTE, workPoint)
    minDCF, rbfSVM_FNRs, rbfSVM_FPRs = Util.compute_minDCF(LTE, calibrated_scores_rbfSVM, workPoint)
    print(f"Calibrated - PCA - rbfSVM DCF: {DCF}")
    print(f"Calibrated - PCA - rbfSVM minDCF: {minDCF}")

    # Fusion
    scores = numpy.vstack([gmm_scores, polySVM_scores])
    _, calibrated_scores_fusion, calibrated_scores_labels = Util.score_calibration(scores, LTE, K, "Fusion",
                                                                                   scaled_workPoint,
                                                                                   "orange")
    predicted = (calibrated_scores_fusion > t).astype(int)
    _, DCF = Util.evaluate(predicted, calibrated_scores_labels, workPoint)
    minDCF, fusion_FNRs, fusion_FPRs = Util.compute_minDCF(calibrated_scores_labels, calibrated_scores_fusion,
                                                           workPoint)
    print(f"Calibrated - PCA - Fusion DCF: {DCF}")
    print(f"Calibrated - PCA - Fusion minDCF: {minDCF}")

    # Bayes error plot comparing the fusion model and those that compose it
    Plots.new_figure()
    Util.bayes_error_calibration_evaluation(calibrated_scores_gmm, LTE, gmmClassifier, "blue")
    Util.bayes_error_calibration_evaluation(calibrated_scores_polySVM, LTE, polySVM, "red")
    Util.bayes_error_calibration_evaluation(calibrated_scores_rbfSVM, LTE, rbfSVM, "green")
    Util.bayes_error_calibration_evaluation(calibrated_scores_fusion, calibrated_scores_labels, "Fusion", "orange")
    Plots.plot_vertical_line(numpy.log(scaled_workPoint.pi / (1 - scaled_workPoint.pi)))
    Plots.show_plot()

    fig = Plots.new_figure()
    Plots.plot_simple_plot_no_show(fig, gmm_FPRs, gmm_FNRs, "False Positive Ratio", "False Negative Ratio",
                                   "blue", gmmClassifier.__str__(), "DET Plot", "log", "log")
    Plots.plot_simple_plot_no_show(fig, polySVM_FPRs, polySVM_FNRs, "False Positive Ratio", "False Negative Ratio",
                                   "red", polySVM.__str__(), "DET Plot", "log", "log")
    Plots.plot_simple_plot_no_show(fig, rbfSVM_FPRs, rbfSVM_FNRs, "False Positive Ratio", "False Negative Ratio",
                                   "green", rbfSVM.__str__(), "DET Plot", "log", "log")
    Plots.plot_simple_plot_no_show(fig, fusion_FPRs, fusion_FNRs, "False Positive Ratio", "False Negative Ratio",
                                   "orange", "Fusion", "DET Plot", "log", "log")
    Plots.show_plot()
    