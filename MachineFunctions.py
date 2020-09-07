


#https://machinelearningmastery.com/compare-machine-learning-algorithms-python-scikit-learn/
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
class Machine:


    def __init__(X,Y):
        # prepare configuration for cross validation test harness
        seed = 2020
        # prepare models
        self.X=X
        self.Y=Y
        self.models = []
        self.results = []
        self.names = []
        self.models.append(('LR', LogisticRegression()))
        self.models.append(('LDA', LinearDiscriminantAnalysis()))
        self.models.append(('KNN', KNeighborsClassifier()))
        self.models.append(('CART', DecisionTreeClassifier()))
        self.models.append(('NB', GaussianNB()))
        self.models.append(('SVM', SVC()))

    def evaluate():
        results = []
        names = []
        scoring = 'accuracy'
        for name, model in models:
            kfold = model_selection.KFold(n_splits=10, random_state=seed)
            cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
            results.append(cv_results)
            names.append(name)
