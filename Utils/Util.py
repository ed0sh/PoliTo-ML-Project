import math

import numpy
import scipy.special
from Classifiers import ClassifiersInterface
import Classifiers
from Utils import Preproccessing, Plots


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
    std = numpy.sqrt(numpy.diag(C))
    return C / numpy.outer(std, std)


def evaluateCorrelation(D: numpy.array, threshold: float):
    correlation_matrix = dataCorrelationMatrix(D)
    featurePairsOver = (numpy.abs(correlation_matrix - numpy.diag(numpy.ones(D.shape[0]))) > threshold).astype(
        int).sum() / 2
    meanCorrelation = correlation_matrix.sum() / correlation_matrix.shape[0] ** 2
    return featurePairsOver.astype(int), meanCorrelation


def evaluateClassCorrelation(D: numpy.array, L: numpy.array, threshold: float):
    meanCorrelations = []
    featurePairsOver = []
    for label in numpy.unique(L):
        pairs, mean = evaluateCorrelation(D[:, L == label], threshold)
        meanCorrelations.append(mean)
        featurePairsOver.append(pairs)
    return featurePairsOver, meanCorrelations


def within_class_covariance(D: numpy.array, N: int):
    return dataCovarianceMatrix(D)[0] * D.size / N


def within_classes_covariance_matrix(D, L):
    Sw = numpy.zeros((D.shape[0], D.shape[0]))
    for c in numpy.unique(L):
        Sw += within_class_covariance(D[:, L == c], D.shape[1])

    return Sw / D.shape[0]


def between_classes_covariance_matrix(D, L):
    Sb = numpy.zeros((D.shape[0], D.shape[0]))
    mu = vcol(D.mean(axis=1))

    for c in numpy.unique(L):
        Dc = D[:, L == c]
        muc = vcol(Dc.mean(axis=1))
        mu_diff = muc - mu
        Sb += numpy.dot(mu_diff, mu_diff.T) * Dc.shape[1]

    return Sb / D.shape[1]


def generalized_eig_problem(Sb, Sw, m):
    s, U = scipy.linalg.eigh(Sb, Sw)
    return U[:, ::-1][:, 0:m]


def joint_diagonalizaton(Sb, Sw, m):
    Uw, sw, _ = numpy.linalg.svd(Sw)
    P1 = numpy.dot(Uw * vrow(1.0 / (sw ** 0.5)), Uw.T)
    Sbt = numpy.dot(P1, numpy.dot(Sb, P1.T))

    Ub, _, _ = numpy.linalg.svd(Sbt)
    P2 = Ub[:, 0:m]
    return numpy.dot(P1.T, P2)


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


def compute_minDCF(LTE, SPost, workPoint):
    idx = numpy.argsort(SPost.ravel())
    sortL = LTE[idx]
    MinDCF = 100
    FNRs = []
    FPRs = []
    startingMatrix = confusion_matrix(LTE, Discriminant_ratio(-math.inf, SPost))
    for val in sortL:
        if val == 0:
            startingMatrix[0][0] = startingMatrix[0][0] + 1
            startingMatrix[1][0] = startingMatrix[1][0] - 1
        else:
            startingMatrix[0][1] = startingMatrix[0][1] + 1
            startingMatrix[1][1] = startingMatrix[1][1] - 1
        FNRs.append(startingMatrix[0][1] / (startingMatrix[0][1] + startingMatrix[1][1]))
        FPRs.append(startingMatrix[1][0] / (startingMatrix[1][0] + startingMatrix[0][0]))

        _, tempDCF = Compute_DCF(startingMatrix, workPoint)
        if (tempDCF < MinDCF):
            MinDCF = tempDCF
    return MinDCF, FNRs, FPRs


def Discriminant_ratio(threshold, SPost):
    res = numpy.copy(SPost)
    res[SPost > threshold] = 1
    res[SPost <= threshold] = 0
    return (SPost > threshold).astype(int)


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


def k_folds(DTR: numpy.array, LTR: numpy.array, K: int, modelObject: ClassifiersInterface, workPoint: WorkPoint, seed=0):
    d_folds, l_folds = split_k_folds(DTR, LTR, K, seed)
    DCFs = []
    error_rates = []
    scores = []
    labels = []
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

        # If we performed class-re-balancing, save a state to re-perform it again after updating the dataset
        svm_rebalanced = False
        if isinstance(modelObject, Classifiers.LinearSVM.LinearSVM) or isinstance(modelObject,
                                                                                  Classifiers.KernelSVM.KernelSVM):
            if modelObject.balanced_C is not None:
                svm_rebalanced = True

        # Update model dataset
        modelObject.update_dataset(data_train_set, labels_train_set)

        # Re-apply class balancing
        if (isinstance(modelObject, Classifiers.LinearSVM.LinearSVM) or isinstance(modelObject,
                                                                                   Classifiers.KernelSVM.KernelSVM)) and svm_rebalanced:
            modelObject.rebalance(workPoint)

        modelObject.train()
        predicted = modelObject.classify(data_test_set, workPoint)
        err_rate, DCF = evaluate(predicted, labels_test_set, workPoint)
        error_rates.append(err_rate)
        DCFs.append(DCF)
        scores.extend(modelObject.get_scores())
        labels.extend(labels_test_set)

    mean_err_rate = numpy.array(error_rates).mean()
    mean_DCF = numpy.array(DCFs).mean()
    minDCF, FNRs, FPRs = compute_minDCF(numpy.array(labels), numpy.array(scores), workPoint)
    return mean_err_rate, mean_DCF, minDCF, (numpy.array(scores), numpy.array(labels)), (FNRs, FPRs)


def evaluate(PLabel: numpy.array, LTE: numpy.array, workPoint: WorkPoint):
    errRate = ((LTE != PLabel).astype(int).sum() / LTE.shape[0]) * 100
    matrix = confusion_matrix(LTE, PLabel)
    _, DCF = Compute_DCF(matrix, workPoint)
    return errRate, DCF


def logpdf_GMM(X: numpy.array, gmm: numpy.array):
    S = []
    for w, mu, C in gmm:
        lod_dens = logpdf_GAU_ND(X, mu, C)
        logPrior = numpy.log(w)
        S.append(lod_dens + logPrior)
    S = numpy.vstack(S)
    logdens = scipy.special.logsumexp(S, axis=0)
    return logdens


def EM(X, gmm, psi, diagonal_cov, tied_cov):
    lls = []
    delta = 1e-6

    P = E_step(X, gmm)
    gmm = M_step(P, X, psi, diagonal_cov, tied_cov)
    ll = logpdf_GMM(X, gmm).sum()
    lls.append(ll)

    i = 0

    while i == 0 or ll - last_ll > delta:
        i += 1
        last_ll = ll
        P = E_step(X, gmm)
        gmm = M_step(P, X, psi, diagonal_cov, tied_cov)
        ll = logpdf_GMM(X, gmm).sum()
        lls.append(ll)

    return gmm, lls, (ll / X.shape[1])


def E_step(X, gmm):
    SJoint = []
    for w, mu, C in gmm:
        lod_dens = logpdf_GAU_ND(X, mu, C)
        logPrior = numpy.log(w)
        SJoint.append(lod_dens + logPrior)
    SJoint = numpy.vstack(SJoint)
    logP = SJoint - vrow(scipy.special.logsumexp(SJoint, axis=0))

    P = numpy.exp(logP)
    return P


def M_step(P, X, psi, diagonal_cov, tied_cov):
    gmm_updated = []
    Zg_tot = P.sum()
    Zgs = []

    for g in P:
        g = vrow(g)

        Z_g = g.sum()
        Zgs.append(Z_g)

        F_g = vcol((X * g).sum(1))
        S_g = numpy.dot((g * X), X.T)

        mu_g = F_g / Z_g
        Sigma_g = (S_g / Z_g) - numpy.dot(mu_g, mu_g.T)

        if diagonal_cov:
            Sigma_g = Sigma_g * numpy.eye(Sigma_g.shape[1])

        Sigma_g = eigen_constraint(Sigma_g, psi)

        w_g = Z_g / Zg_tot
        gmm_updated.append((w_g, mu_g, Sigma_g))

    if tied_cov:
        Sigma_tied = sum([w * C for w, mu, C in gmm_updated])
        Sigma_tied = eigen_constraint(Sigma_tied, psi)
        gmm_updated = [(w, mu, Sigma_tied) for w, mu, _ in gmm_updated]

    return gmm_updated


def LBG(X, gmm, alpha, max_g, psi, diagonal_cov, tied_cov):
    g = 1
    gmm = [(gmm[0][0], gmm[0][1], eigen_constraint(gmm[0][2], psi))]

    ll_mean = 0
    all_lls = []

    while g < max_g:
        new_gmm = []
        for gmm_comp in gmm:
            w, mu, Sigma = gmm_comp

            U, s, Vh = numpy.linalg.svd(Sigma)
            d = U[:, 0:1] * s[0] ** 0.5 * alpha

            new_gmm.append((w / 2, mu + d, Sigma))
            new_gmm.append((w / 2, mu - d, Sigma))

        gmm, lls, ll_mean = EM(X, new_gmm, psi, diagonal_cov, tied_cov)
        g *= 2
        all_lls.extend(lls)

    return gmm, all_lls, ll_mean


def eigen_constraint(Sigma, psi):
    U, s, _ = numpy.linalg.svd(Sigma)
    s[s < psi] = psi
    Sigma = numpy.dot(U, vcol(s) * U.T)
    return Sigma


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
                attrs = vcol(numpy.array([float(i) for i in attrs]))
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


def evaluate_model(DTR: numpy.array, LTR: numpy.array, PCA_values: list, K: int, modelObject: ClassifiersInterface,
                   scaled_workPoint: WorkPoint):
    print("PCA\t|\tminDCF")

    minDCF_values = []
    DCF_values = []
    for m in PCA_values:
        if m is not None:
            reduced_DTR = Preproccessing.PCA(DTR, m)[0]
        else:
            reduced_DTR = DTR.copy()
            m = "No"

        # If we performed class-re-balancing, save a state to re-perform it again after updating the dataset
        svm_rebalanced = False
        if isinstance(modelObject, Classifiers.LinearSVM.LinearSVM) \
                or isinstance(modelObject, Classifiers.KernelSVM.KernelSVM):
            if modelObject.balanced_C is not None:
                svm_rebalanced = True

        modelObject.update_dataset(reduced_DTR, LTR)

        # Re-apply class balancing
        if (isinstance(modelObject, Classifiers.LinearSVM.LinearSVM) or isinstance(modelObject,
                                                                                   Classifiers.KernelSVM.KernelSVM)) \
                and svm_rebalanced:
            modelObject.rebalance(scaled_workPoint)

        # Apply feature expansion after changing the dataset
        if isinstance(modelObject, Classifiers.LogisticRegression.LogRegClass):
            if modelObject.quadratic:
                modelObject.feature_expansion_inplace()
                reduced_DTR = modelObject.DTR

            modelObject.optimize_lambda_inplace(scaled_workPoint)
            print(f"λ: {modelObject.lam}")

        error, DCF, minDCF, _, _ = k_folds(reduced_DTR, LTR, K, modelObject, scaled_workPoint)
        minDCF = round(minDCF, 3)
        DCF = round(DCF, 3)
        print(f"{m}\t|\t{minDCF}")
        minDCF_values.append(minDCF)
        DCF_values.append(DCF)
    return minDCF_values, DCF_values


def bayes_error_calibration_evaluation_k_folds(DTR, LTR, PCA_value, K, modelObject, scaled_workPoint, color):
    if PCA_value is not None:
        DTR = Preproccessing.PCA(DTR, PCA_value)[0]
    _, _, _, (scores, labels), _ = k_folds(DTR, LTR, K, modelObject, scaled_workPoint)

    bayes_error_calibration_evaluation(scores, labels, modelObject, color)


def bayes_error_calibration_evaluation(scores, labels, modelObject, color):
    effPriorLogOdds = numpy.linspace(-3, 3, 100)
    DCFs = []
    minDCFs = []
    for p in effPriorLogOdds:
        eff_pi = 1 / (1 + numpy.exp(-p))

        conf_matrix = confusion_matrix(labels, (scores > -p).astype(int).ravel())
        DCF = Compute_Anormalized_DCF(conf_matrix, eff_pi, 1, 1)
        DCF = Compute_Normalized_DCF(DCF, eff_pi, 1, 1)
        DCFs.append(DCF)
        minDCFs.append(compute_minDCF(labels, scores, WorkPoint(eff_pi, 1, 1))[0])

    Plots.plot_bayes_error_plot_no_show(effPriorLogOdds, DCFs, minDCFs, modelObject.__str__(), color)


def score_calibration_k_folds(DTR, LTR, PCA_value, K, modelObject, scaled_workPoint, color):
    if PCA_value is not None:
        DTR = Preproccessing.PCA(DTR, PCA_value)[0]

    # Collect the pooled scores and their relative labels from k_folds
    _, _, _, (scores, labels), _ = k_folds(DTR, LTR, K, modelObject, scaled_workPoint)

    return score_calibration(scores, labels, K, modelObject, scaled_workPoint, color)


def score_calibration(scores: numpy.array, labels, K, modelObject, scaled_workPoint, color):
    if len(scores.shape) == 1:
        scores = vrow(scores)

    # Shuffle the scores
    numpy.random.seed(4131)
    idx = numpy.random.permutation(scores.shape[1])
    shuffled_scores = scores[:, idx]
    shuffled_labels = labels[idx]

    logReg = Classifiers.LogisticRegression.LogRegClass(shuffled_scores, shuffled_labels,
                                                        l=1e1,
                                                        prior=scaled_workPoint.effective_prior())

    # Find the best lambda according to the minimization of the distance between DCF and minDCF
    selectedLambda = 1e1
    lambdas = [1e1, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    bestMinDCF = 10

    for lam in lambdas:
        logRegObj = Classifiers.LogisticRegression.LogRegClass(shuffled_scores, shuffled_labels, lam, logReg.prior)
        _, DCF, minDCF, _, _ = k_folds(shuffled_scores, shuffled_labels, K, logRegObj, scaled_workPoint)

        if (DCF - minDCF) < bestMinDCF:
            selectedLambda = lam
            bestMinDCF = (DCF - minDCF)
    logReg.lam = selectedLambda

    _, _, _, (new_scores, calibrated_scores_labels), _ = k_folds(shuffled_scores, shuffled_labels, K, logReg, scaled_workPoint, seed=134)
    if modelObject is not None:
        bayes_error_calibration_evaluation(new_scores, calibrated_scores_labels, modelObject, color)

    calibration_model = Classifiers.LogisticRegression.LogRegClass(shuffled_scores, shuffled_labels, selectedLambda, logReg.prior)
    calibration_model.train()

    return calibration_model, new_scores, calibrated_scores_labels


def DET_plot(DTR, LTR, PCA_value, K, modelObject, scaled_workPoint, fig, color):
    if PCA_value is not None:
        DTR = Preproccessing.PCA(DTR, PCA_value)[0]
    _, _, _, _, (FNRs, FPRs) = k_folds(DTR, LTR, K, modelObject, scaled_workPoint)
    Plots.plot_simple_plot_no_show(fig, FPRs, FNRs,
                                   "False Positive Ratio",
                                   "False Negative Ratio",
                                   color,
                                   modelObject.__str__(),
                                   "DET Plot",
                                   "log",
                                   "log"
                                   )


def svm_cross_val_graphs(
        K_svm_vec: numpy.array,
        C_vec: numpy.array,
        DTR: numpy.array,
        LTR: numpy.array,
        PCA_values: list,
        K: int,
        scaled_workPoint: WorkPoint,
        rebalanced: bool,
        colors: list,
        svm_type_label: str,
        kernel_type: str = None,
        c=0.0, d=2, gamma=1):

    fig = Plots.new_figure()
    for ki, K_svm in enumerate(K_svm_vec):
        C_results = []
        for C in C_vec:
            print(f"K: {K_svm}, C: {C}")

            if kernel_type == "poly" or kernel_type == "rbf":
                modelSVM = Classifiers.KernelSVM.KernelSVM(DTR, LTR, C, K_svm, kernel_type=kernel_type, d=d, c=c,
                                                           gamma=gamma)
            else:
                modelSVM = Classifiers.LinearSVM.LinearSVM(DTR, LTR, C, K_svm)

            if rebalanced:
                modelSVM.rebalance(scaled_workPoint)
            minDCF_values = evaluate_model(DTR, LTR, PCA_values, K, modelSVM, scaled_workPoint)[0]
            C_results.append(minDCF_values)

        print(f"K: {K_svm}")
        print(C_results)
        for i, m in enumerate(PCA_values):
            Plots.plot_simple_plot_no_show(
                fig,
                C_vec,
                numpy.array(C_results)[:, i],
                x_label="C",
                y_label=f"minDCF",
                color=colors[(ki * len(PCA_values)) + i],
                label=f"PCA={m} - K={K_svm}",
                title=f"{svm_type_label}",
                x_scale="log",
                y_scale="linear"
            )
    Plots.show_plot()


def gmm_grid_search_one_prop(DTR: numpy.array, LTR: numpy.array,
                             prop_c0_vec: numpy.array, prop_c1_vec: numpy.array,
                             prop_name: str,
                             params_gmm_target: dict,
                             params_gmm_non_target: dict,
                             PCA_values: list,
                             K: int,
                             scaled_workPoint: WorkPoint):
    prop_minDCFs = {}
    for prop_c0 in prop_c0_vec:
        params_gmm_non_target[prop_name] = prop_c0

        prop_c1_minDCFs = []
        for prop_c1 in prop_c1_vec:
            print(f"----- {prop_name}_c0 = {prop_c0}, {prop_name}_c1 = {prop_c1} -----")
            params_gmm_target[prop_name] = prop_c1

            gmm_classifier = Classifiers.GMMClassifier.GMMClassifier(DTR, LTR, params_gmm_target, params_gmm_non_target)
            minDCFs = evaluate_model(gmm_classifier.DTR, LTR, PCA_values, K, gmm_classifier, scaled_workPoint)[0]

            prop_c1_minDCFs.append(minDCFs[0])
        prop_minDCFs[prop_c0] = prop_c1_minDCFs

    return prop_minDCFs


def get_sigma_type_as_string(diagonal: bool, tied: bool):
    if diagonal and tied:
        return "Tied Diagonal"
    elif diagonal:
        return "Diagonal"
    elif tied:
        return "Tied"
    else:
        return "Full"


def evaluate_model_fusion(DTR: numpy.array, LTR: numpy.array,
                          PCA_value: int, K: int,
                          modelObject1: ClassifiersInterface,
                          modelObject2: ClassifiersInterface,
                          modelObject3: ClassifiersInterface,
                          scaled_workPoint: WorkPoint):
    print("PCA\t|\tminDCF\t|\tDCF")

    if PCA_value is not None:
        reduced_DTR = Preproccessing.PCA(DTR, PCA_value)[0]
    else:
        reduced_DTR = DTR.copy()
        PCA_value = "No"

    modelObject1.update_dataset(reduced_DTR, LTR)
    modelObject2.update_dataset(reduced_DTR, LTR)
    if modelObject3 is not None:
        modelObject3.update_dataset(reduced_DTR, LTR)

    _, _, _, (scores1, labels1), _ = k_folds(reduced_DTR, LTR, K, modelObject1, scaled_workPoint, seed=564)
    _, _, _, (scores2, labels2), _ = k_folds(reduced_DTR, LTR, K, modelObject2, scaled_workPoint, seed=564)

    if modelObject3 is not None:
        _, _, _, (scores3, labels3), _ = k_folds(reduced_DTR, LTR, K, modelObject3, scaled_workPoint, seed=564)
        scores = numpy.vstack([scores1, scores2, scores3])
    else:
        scores = numpy.vstack([scores1, scores2])

    _, new_scores, calibrated_scores_labels = score_calibration(scores, labels1, K, "Fusion", scaled_workPoint,
                                                                "orange")

    t = - numpy.log(scaled_workPoint.pi / (1 - scaled_workPoint.pi))

    # Fusion model calibrated DCF and minDCF
    predicted = (new_scores > t).astype(int)
    _, DCF = evaluate(predicted, calibrated_scores_labels, scaled_workPoint)
    minDCF = compute_minDCF(calibrated_scores_labels, new_scores, scaled_workPoint)[0]

    minDCF = round(minDCF, 3)
    DCF = round(DCF, 3)

    print(f"{PCA_value}\t|\t{minDCF}\t|\t{DCF}")

    return minDCF, DCF


def DET_plot_fusion(DTR: numpy.array, LTR: numpy.array,
                    PCA_value: int, K: int,
                    modelObject1: ClassifiersInterface,
                    modelObject2: ClassifiersInterface,
                    modelObject3: ClassifiersInterface,
                    scaled_workPoint: WorkPoint,
                    fig, color):

    if PCA_value is not None:
        reduced_DTR = Preproccessing.PCA(DTR, PCA_value)[0]
    else:
        reduced_DTR = DTR.copy()
        PCA_value = "No"

    modelObject1.update_dataset(reduced_DTR, LTR)
    modelObject2.update_dataset(reduced_DTR, LTR)
    if modelObject3 is not None:
        modelObject3.update_dataset(reduced_DTR, LTR)

    _, _, _, (scores1, labels1), _ = k_folds(reduced_DTR, LTR, K, modelObject1, scaled_workPoint, seed=564)
    _, _, _, (scores2, labels2), _ = k_folds(reduced_DTR, LTR, K, modelObject2, scaled_workPoint, seed=564)

    if modelObject3 is not None:
        _, _, _, (scores3, labels3), _ = k_folds(reduced_DTR, LTR, K, modelObject3, scaled_workPoint, seed=564)
        scores = numpy.vstack([scores1, scores2, scores3])
    else:
        scores = numpy.vstack([scores1, scores2])

    _, new_scores, calibrated_scores_labels = score_calibration(scores, labels1, K, None, scaled_workPoint,None)
    _, FNRs, FPRs = compute_minDCF(calibrated_scores_labels, new_scores, scaled_workPoint)

    Plots.plot_simple_plot_no_show(fig, FPRs, FNRs,
                                   "False Positive Ratio",
                                   "False Negative Ratio",
                                   color,
                                   "Fusion",
                                   "DET Plot",
                                   "log",
                                   "log"
                                   )


def svm_hyperparam_eval_graphs(
        K_svm_vec: numpy.array,
        C_vec: numpy.array,
        DTR: numpy.array,
        LTR: numpy.array,
        DTE: numpy.array,
        LTE: numpy.array,
        PCA_values: list,
        scaled_workPoint: WorkPoint,
        rebalanced: bool,
        colors: list,
        svm_type_label: str,
        kernel_type: str = None,
        c=0.0, d=2, gamma=1):

    fig = Plots.new_figure()
    for ki, K_svm in enumerate(K_svm_vec):
        C_results = []
        for C in C_vec:
            print(f"K: {K_svm}, C: {C}")

            if kernel_type == "poly" or kernel_type == "rbf":
                modelSVM = Classifiers.KernelSVM.KernelSVM(DTR, LTR, C, K_svm, kernel_type=kernel_type, d=d, c=c,
                                                           gamma=gamma)
            else:
                modelSVM = Classifiers.LinearSVM.LinearSVM(DTR, LTR, C, K_svm)

            minDCF_values = []

            for PCA_value in PCA_values:
                PCAed_DTR, _, P = Preproccessing.PCA(DTR, PCA_value)
                PCAed_DTE = numpy.dot(P.T, DTE)

                modelSVM.update_dataset(PCAed_DTR, LTR)

                if rebalanced:
                    modelSVM.rebalance(scaled_workPoint)

                modelSVM.train()
                modelSVM.classify(PCAed_DTE, scaled_workPoint)
                modelSVM_scores = vrow(modelSVM.get_scores())

                minDCF = compute_minDCF(LTE, modelSVM_scores, scaled_workPoint)[0]
                print(f"\tminDCF: {minDCF}")
                minDCF_values.append(minDCF)

            C_results.append(minDCF_values)

        print(f"K: {K_svm}")
        print(C_results)
        for i, m in enumerate(PCA_values):
            Plots.plot_simple_plot_no_show(
                fig,
                C_vec,
                numpy.array(C_results)[:, i],
                x_label="C",
                y_label=f"minDCF",
                color=colors[(ki * len(PCA_values)) + i],
                label=f"PCA={m} - K={K_svm}",
                title=f"{svm_type_label}",
                x_scale="log",
                y_scale="linear"
            )
    Plots.show_plot()


def gmm_hyperparam_eval_one_prop(DTR: numpy.array, LTR: numpy.array, DTE: numpy.array, LTE: numpy.array,
                             prop_c0_vec: numpy.array, prop_c1_vec: numpy.array,
                             prop_name: str,
                             params_gmm_target: dict,
                             params_gmm_non_target: dict,
                             PCA_values: list,
                             scaled_workPoint: WorkPoint):
    prop_minDCFs = {}
    for prop_c0 in prop_c0_vec:
        params_gmm_non_target[prop_name] = prop_c0

        prop_c1_minDCFs = []
        for prop_c1 in prop_c1_vec:
            print(f"{prop_name}_c0 = {prop_c0}, {prop_name}_c1 = {prop_c1}")
            params_gmm_target[prop_name] = prop_c1

            gmm_classifier = Classifiers.GMMClassifier.GMMClassifier(DTR, LTR, params_gmm_target, params_gmm_non_target)

            for PCA_value in PCA_values:
                PCAed_DTR, _, P = Preproccessing.PCA(DTR, PCA_value)
                PCAed_DTE = numpy.dot(P.T, DTE)

                gmm_classifier.update_dataset(PCAed_DTR, LTR)
                gmm_classifier.train()
                gmm_classifier.classify(PCAed_DTE, scaled_workPoint)
                modelSVM_scores = vrow(gmm_classifier.get_scores())

                minDCF = round(compute_minDCF(LTE, modelSVM_scores, scaled_workPoint)[0], 3)
                print(f"\tminDCF: {minDCF}")
                prop_c1_minDCFs.append(minDCF)

        prop_minDCFs[prop_c0] = prop_c1_minDCFs

    return prop_minDCFs