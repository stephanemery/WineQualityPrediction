from abc import ABC, abstractmethod

class Machine(ABC):
    def __init__(self, name):
        self.name = name
        self.trained = False
        self.score = 0.0

    @abstractmethod
    def train(self, X, y):
        """
            Train the machine
        """
        pass

    @abstractmethod
    def test(self, X, y):
        """
            Test the machine
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
            Predict output from input values
        """
        pass

    def __str__(self):
        """
            Display the machine's informations
        """
        txt = self.name
        if not self.trained :
            txt += '\r\n\tAlgorithm not trained !'
        else :
            txt += '\r\n\tScore : ' + str(self.score)

        return txt