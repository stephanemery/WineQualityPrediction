from .Machine import Machine
from sklearn.neighbors import KNeighborsRegressor

class KNN(Machine):
    def __init__(self):
        super().__init__('KNN Regressor')
        self.reg = KNeighborsRegressor(10)

    def train(self, X, y):
        # Fit data
        self.reg.fit(X, y)

        self.trained = True

    def test(self, X, y):
        if not self.trained:
            print('The algorithm is not trained !')
            return
        
        # Compute the score
        self.score = self.reg.score(X,y)

    def predict(self, X):
        if not self.trained:
            print('The algorithm is not trained !')
            return
        
        return self.reg.predict(X)

