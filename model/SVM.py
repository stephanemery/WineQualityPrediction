from .Machine import Machine
from sklearn.svm import LinearSVR


class SVM(Machine):
    def __init__(self, epsilon, C):
        super().__init__("SVM Regressor")
        self.reg = LinearSVR(epsilon=epsilon, C=C)

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
