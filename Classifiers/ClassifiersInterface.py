import numpy


class ClassifiersInterface:
    def train(self):
        pass

    def classify(self, DTE: numpy.array, workpoint) -> numpy.array:
        pass

    def update_dataset(self, DTR: numpy.array, LTR: numpy.array):
        pass

    def get_scores(self) -> numpy.array:
        if self.scores is None:
            return RuntimeError("Scores not defined")
        return self.scores