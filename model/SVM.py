from .Machine import Machine
from sklearn.svm import LinearSVR


class SVM(Machine):
    """
    Support Vector Machine model
    
    Parameters
    ========
    
    epsilon : float
        Epsilon parameter in the epsilon-insensitive loss function. Note that the value of this parameter depends on the scale of the target variable y. If unsure, set *epsilon=0*.
        
    C : float
        Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive.
    """

    def __init__(self, epsilon, C):
        super().__init__("SVM Regressor")
        self.reg = LinearSVR(epsilon=epsilon, C=C)

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
