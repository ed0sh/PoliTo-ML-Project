import numpy

from Utils import Util


class ClassifiersInterface:
    def train(self):
        pass

    def classify(self, DTE: numpy.array, workpoint: Util.WorkPoint) -> numpy.array:
        pass

    def update_dataset(self, DTR: numpy.array, LTR: numpy.array):
        pass

    def get_scores(self) -> numpy.array:
        if self.scores is None:
            return RuntimeError("Scores not defined")
        return self.scores