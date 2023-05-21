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

    util.plot_scatter(DTR, LTR)
    util.plot_hists(DTR, LTR)

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

    print("----- Naive MVG -----")
    logNaiveMVG.train()
    P = logNaiveMVG.classify(DTE)
    SPost = P.argmax(axis=0)
    error, DCF = util.evaluate(SPost, LTE, workPoint)
    print(f"Error rate : {error} \nNormalized DCF : {DCF}")

    print("----- NaiveMVG with PCA -----")
    D = util.PCA(DTR, 4)
    logNaiveMVG = NaiveMVGClassifier(D, LTR, workPoint.pi)
    logNaiveMVG.train()
    P = logNaiveMVG.classify(util.PCA(DTE, 4))
    SPost = P.argmax(axis=0)
    error, DCF = util.evaluate(SPost, LTE, workPoint)
    print(f"Error rate : {error} \nNormalized DCF : {DCF}")

    print("----- TiedMVG -----")
    logTiedMVG.train()
    P = logTiedMVG.classify(DTE)
    SPost = P.argmax(axis=0)
    error, DCF = util.evaluate(SPost, LTE, workPoint)
    print(f"Error rate : {error} \nNormalized DCF : {DCF}")

    print("----- log Regression -----")
    logReg.train()
    PLabels = logReg.classify(DTE)
    error, DCF = util.evaluate(PLabels, LTE, workPoint)
    print(f"Error rate : {error} \nNormalized DCF : {DCF}")

    util.k_folds(DTR, LTR, DTR.shape[0], MVGClassifier, workPoint.pi)
