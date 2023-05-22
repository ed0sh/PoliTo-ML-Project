import numpy


class ClassifiersInterface:
    def train(self):
        pass

    def classify(self, DTE: numpy.array) -> numpy.array:
        pass

    def update_dataset(self, DTR: numpy.array, LTR: numpy.array):
        pass
