import numpy


class ClassifiersInterface:
    def train(self):
        pass

    def classify(self, DTE: numpy.array) -> numpy.array:
        pass
