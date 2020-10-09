.. image:: https://travis-ci.com/stephanemery/WineQualityPrediction.svg?branch=master
    :target: https://travis-ci.com/stephanemery/WineQualityPrediction
.. image:: https://coveralls.io/repos/github/stephanemery/WineQualityPrediction/badge.svg?branch=master
    :target: https://coveralls.io/github/stephanemery/WineQualityPrediction?branch=master
.. image:: https://img.shields.io/badge/docs-latest-orange.svg
    :target: https://stephanemery.github.io/WineQualityPrediction/
.. image:: https://img.shields.io/badge/github-project-0000c0.svg
    :target: https://github.com/stephanemery/WineQualityPrediction
.. image:: https://img.shields.io/github/contributors/stephanemery/WineQualityPrediction.svg
    :target: https://github.com/stephanemery/WineQualityPrediction/graphs/contributors
.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :target: https://github.com/stephanemery/WineQualityPrediction/blob/master/LICENSE
.. image:: https://img.shields.io/github/issues/stephanemery/WineQualityPrediction
    :target: https://github.com/stephanemery/WineQualityPrediction/issues
  
Wine quality prediction from its physicochemical properties
===========================================================

In this project, we try to predict the wine quality from its physicochemical properties.

Methods used :
--------------

1. Preprocessing
    * Drop NaN values
    * Outliers removal
    * Normalize data (Standard or MinMax scaler)
  
2. Models
    * Multi-Linear Regression
    * SVM Regressor
    * KNN Regressor

Results
-------

.. code:: sh
   
   Preprocessing done !
   Multi-Linear Regression
           Score : 0.252751
   KNN Regressor
           Score : 0.178368
   SVM Regressor
           Score : 0.247992

Check the documentation_ to reproduce the results.

.. _documentation: https://stephanemery.github.io/WineQualityPrediction/
