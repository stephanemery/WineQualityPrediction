from .Machine import Machine
from sklearn.linear_model import LinearRegression


class MultiLinearRegression(Machine):
    def __init__(self):
        super().__init__("Multi-Linear Regression")
        self.reg = LinearRegression()

    def train(self, X, y):
        # Fit data
        self.reg.fit(X, y)

        self.trained = True

    def test(self, X, y):
        if not self.trained:
            print("The algorithm is not trained !")
            return

        # Compute the score
        self.score = self.reg.score(X, y)

    def predict(self, X):
        if not self.trained:
            print("The algorithm is not trained !")
            return

        return self.reg.predict(X)
