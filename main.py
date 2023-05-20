import numpy
import util
from Classifier import logMVG , linearMVG , logTiedMVG , logNaiveMVG
from Project.Classifiers.LogisticRegression import LogRegClass

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
        D , L = numpy.hstack(DList), numpy.array(labelsList, dtype=numpy.int32)
        return D, L

if __name__ == '__main__':

    # Load the dataset
    (DTR, LTR) = readfile('data/Train.csv')
    (DTE, LTE) = readfile('data/Test.csv')

    _, P = logMVG(DTR, LTR, DTE)
    SPost = P.argmax(axis=0)
    error = (SPost != LTE).sum() / LTE.shape[0] * 100
    print(error)

    _, P = logNaiveMVG(DTR, LTR, DTE)
    SPost = P.argmax(axis=0)
    error = (SPost != LTE).sum() / LTE.shape[0] * 100
    print(error)

    _, P = logTiedMVG(DTR, LTR, DTE)
    SPost = P.argmax(axis=0)
    error = (SPost != LTE).sum() / LTE.shape[0] * 100
    print(error)

    error, w, b = LogRegClass(DTR, LTR, 0.00001).evaluate(DTE, LTE)
    print(error)
    print(LogRegClass(DTR, LTR, 1).confusion_matrix(DTE, LTE))

    util.k_folds(DTR,LTR,DTR.shape[0],logMVG)
