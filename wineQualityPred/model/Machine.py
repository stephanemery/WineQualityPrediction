from abc import ABC, abstractmethod

"""
Parent class of the models
"""

class Machine(ABC):
    def __init__(self, name):
        self.name = name
        self.trained = False
        self.score = 0.0

    @abstractmethod
    def train(self, X, y):
        """
        Train the machine
        
        Parameters
        ==========

        X : numpy.ndarray
                The training data array (NxM).
        y : numpy.ndarray
            The expected output (Nx1)

        """
        pass

    @abstractmethod
    def test(self, X, y):
        """
        Test the machine

        Parameters
        ==========

        X : numpy.ndarray
                The training data array (NxM).
        y : numpy.ndarray
            The expected output (Nx1)

        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Predict outputs from inputs data

        Parameters
        ==========

        X : numpy.ndarray
                The data array (NxM).
        
        Returns
        =======

        : numpy.ndarray
            The result of the prediction
            
        """
        pass

    def __str__(self):
        """
        Display the model's informations
        """
        txt = self.name
        if not self.trained:
            txt += "\r\n\tAlgorithm not trained !"
        else:
            txt += "\r\n\tScore : " + "{:.6f}".format(self.score)

        return txt
