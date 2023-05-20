import numpy


class ClassifiersInterface:
    def train(self):
        pass

    def classify(self, DTE: numpy.array) -> (numpy.array, numpy.array):
        pass

    def evaluate(self, DTE: numpy.array, LTE: numpy.array) -> (float, numpy.array):
        pass
