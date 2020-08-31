import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split #from sklearn.cross_validation import train_test_split


class Machine:

  def __init__(self, filePath='data/winequality-red.csv', algo='linear Regression',debug='False'):
    self.random_state=2020
    self.data = pd.read_csv(filePath)
    self.X= self.data.iloc[:,:-1].values
    self.y= self.data.iloc[:,:-1].values
    #split_dataSet with the same values
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split()
    if debug:      
      self.data.head()
      self.data.describe()
      plt.figure(figsize=(10,10))
      sns.heatmap(self.data.corr(),annot=True,linewidth=0.5,center=0,cmap='coolwarm')
      plt.show()
  def resplit_dataSet(test_size=0.2,random_state=2020):
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.3, random_state = 2020)
    self.random_state=random_state

  def randomForestRegressorModel():
    rfc = RandomForestClassifier(random_state=2020, oob_score=True)
    param_dist = {"n_estimators": [50, 100, 150, 200, 250], 'min_samples_leaf': [1, 2, 4]}
    rfc_gs = GridSearchCV(rfc, param_grid=param_dist, scoring='accuracy', cv=5)
    rfc_gs.fit(self.X_train, self.y_train)
    rfc_gs_rs.best_score_
    importances = rfc_gs_rs.best_estimator_.feature_importances_
    feature_importances = pd.DataFrame(importances,index = wine.columns[:-1], columns=['importance']).sort_values('importance',ascending=False)
    feature_importances.plot(kind='barh')
    pred_rfc_rs = rfc_gs_rs.predict(X_test)
    print (pred_rfc_rs)

