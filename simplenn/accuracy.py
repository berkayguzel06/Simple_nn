import numpy as np

class Accuracy:
    def calculate(self, predictions, y):
        comparisons = self.compare(predictions, y)
        acc = np.mean(comparisons)
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)
        return acc
    
    def calculate_accumulated(self):
        accuracy = self.accumulated_sum / self.accumulated_count
        return accuracy
    
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

class Accuracy_Regression(Accuracy):
    def __init__(self):
        self.precision = None
    def addPrecision(self, y, reinit=False):
        if self.precision == None or reinit:
            self.precision = np.std(y) / 250
    def compare(self, predictions, y):
        return np.absolute(predictions-y) < self.precision
    
class Accuracy_Categorical(Accuracy):
    def addPrecision(self, y):
        pass
    def compare(self, predictions, y):
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y