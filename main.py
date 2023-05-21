import numpy
from matplotlib import pyplot as plt

import util
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
                attrs = util.vcol(numpy.array([float(i) for i in attrs]))
                name = line.split(',')[-1].strip()
                label = hLabels[name]
                DList.append(attrs)
                labelsList.append(label)
            except:
                pass
        D, L = numpy.hstack(DList), numpy.array(labelsList, dtype=numpy.int32)
        return D, L


if __name__ == '__main__':
    # Load the dataset
    (DTR, LTR) = readfile('data/Train.csv')
    (DTE, LTE) = readfile('data/Test.csv')

    Z_DTR = util.Z_Score(DTR)
    Z_DTE = util.Z_Score(DTE)

    util.plot_scatter(util.PCA(DTR,2), LTR)
    util.plot_hists(Z_DTR, LTR)

    workPoint = util.WorkPoint(0.5, 1, 10)

    logMVG = MVGClassifier(DTR, LTR, workPoint.pi)
    logNaiveMVG = NaiveMVGClassifier(DTR, LTR, workPoint.pi)
    logTiedMVG = TiedMVGClassifier(DTR, LTR, workPoint.pi)
    logReg = LogRegClass(DTR, LTR, 0.00001)

    print("----- MVG -----")
    logMVG.train()
    P = logMVG.classify(DTE)
    SPost = P.argmax(axis=0)
    error, DCF = util.evaluate(SPost, LTE, workPoint)
    print(f"Error rate : {error} \nNormalized DCF : {DCF}")

    print("----- MVG With Z-Score -----")
    logMVG = MVGClassifier(Z_DTR, LTR, workPoint.pi)
    logMVG.train()
    P = logMVG.classify(Z_DTE)
    SPost = P.argmax(axis=0)
    error, DCF = util.evaluate(SPost, LTE, workPoint)
    print(f"Error rate : {error} \nNormalized DCF : {DCF}")

    print("----- Naive MVG -----")
    logNaiveMVG.train()
    P = logNaiveMVG.classify(DTE)
    SPost = P.argmax(axis=0)
    error, DCF = util.evaluate(SPost, LTE, workPoint)
    print(f"Error rate : {error} \nNormalized DCF : {DCF}")

    print("----- Naive MVG with ZScore -----")
    logNaiveMVG = NaiveMVGClassifier(Z_DTR, LTR, workPoint.pi)
    logNaiveMVG.train()
    P = logNaiveMVG.classify(Z_DTE)
    SPost = P.argmax(axis=0)
    error, DCF = util.evaluate(SPost, LTE, workPoint)
    print(f"Error rate : {error} \nNormalized DCF : {DCF}")

    print("----- TiedMVG -----")
    logTiedMVG.train()
    P = logTiedMVG.classify(DTE)
    SPost = P.argmax(axis=0)
    error, DCF = util.evaluate(SPost, LTE, workPoint)
    print(f"Error rate : {error} \nNormalized DCF : {DCF}")

    print("----- TiedMVG with ZScore-----")
    logTiedMVG = TiedMVGClassifier(Z_DTR, LTR, workPoint.pi)
    logTiedMVG.train()
    P = logTiedMVG.classify(Z_DTE)
    SPost = P.argmax(axis=0)
    error, DCF = util.evaluate(SPost, LTE, workPoint)
    print(f"Error rate : {error} \nNormalized DCF : {DCF}")


    print("----- log Regression -----")
    logReg.train()
    PLabels = logReg.classify(DTE)
    error, DCF = util.evaluate(PLabels, LTE, workPoint)
    print(f"Error rate : {error} \nNormalized DCF : {DCF}")

    print("----- log Regression with ZScore-----")
    logReg = LogRegClass(Z_DTR, LTR, 0.00001)
    logReg.train()
    PLabels = logReg.classify(Z_DTE)
    error, DCF = util.evaluate(PLabels, LTE, workPoint)
    print(f"Error rate : {error} \nNormalized DCF : {DCF}")

    print("----- log Regression with different lambdas-----")
    for lam in [10 ** x for x in range(-8, 3)]:
        print(f"Actual Lambda : {lam}")
        logReg = LogRegClass(DTR, LTR, lam)
        logReg.train()
        PLabels = logReg.classify(DTE)
        error, DCF = util.evaluate(PLabels, LTE, workPoint)
        print(f"Error rate : {error} \nNormalized DCF : {DCF}")
        print()

    print("----- KFolds logMVG -----")
    error, DCF = util.k_folds(DTR, LTR, DTR.shape[0], MVGClassifier, workPoint.pi, workPoint)
    print(f"Error rate : {error} \nNormalized DCF : {DCF}")
