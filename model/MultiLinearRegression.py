from .Machine import Machine
from sklearn.linear_model import LinearRegression

class MultiLinearRegression(Machine):
    """ 
    Multi-Linear Regression model
    """
    def __init__(self):
        super().__init__("Multi-Linear Regression")
        self.reg = LinearRegression()

    def train(self, X, y):
        """
        Train the model.

        Parameters
        ==========

        X : numpy.ndarray
                The training data array (NxM).
        y : numpy.ndarray
            The expected output (Nx1)

        """
        # Fit data
        self.reg.fit(X, y)

        self.trained = True

    def test(self, X, y):
        """
        Test the model.

        Parameters
        ==========

        X : numpy.ndarray
                The testing data array (NxM).
        y : numpy.ndarray
            The expected output (Nx1)

        """
        if not self.trained:
            print("The algorithm is not trained !")
            return

        # Compute the score
        self.score = self.reg.score(X, y)

    def predict(self, X):
        """
        Predict the output from the data.

        Parameters
        ==========

        X : numpy.ndarray
                The data array (NxM).
        
        Returns
        =======

        : numpy.ndarray
            The result of the prediction
        """
        if not self.trained:
            print("The algorithm is not trained !")
            return

        return self.reg.predict(X)
