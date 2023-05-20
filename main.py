import numpy

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

    logMVG = MVGClassifier(DTR, LTR)
    logNaiveMVG = NaiveMVGClassifier(DTR, LTR)
    logTiedMVG = TiedMVGClassifier(DTR, LTR)
    workPoint = util.WorkPoint(0.5, 1, 10)

    logMVG.train()
    _, P = logMVG.classify(DTE)
    SPost = P.argmax(axis=0)
    error = (SPost != LTE).sum() / LTE.shape[0] * 100
    print(error)

    logNaiveMVG.train()
    _, P = logNaiveMVG.classify(DTE)
    SPost = P.argmax(axis=0)
    error = (SPost != LTE).sum() / LTE.shape[0] * 100
    print(error)

    logTiedMVG.train()
    _, P = logTiedMVG.classify(DTE)
    SPost = P.argmax(axis=0)
    error = (SPost != LTE).sum() / LTE.shape[0] * 100
    print(error)

    error, _ = LogRegClass(DTR, LTR, 0.00001).evaluate(DTE, LTE)
    print(error)
    print(util.Compute_DCF(LogRegClass(DTR, LTR, 1).confusion_matrix(DTE, LTE), workPoint))

    util.k_folds(DTR, LTR, DTR.shape[0], MVGClassifier)
