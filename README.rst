.. image:: https://travis-ci.com/stephanemery/WineQualityPrediction.svg?branch=master
    :target: https://travis-ci.com/stephanemery/WineQualityPrediction
.. image:: https://coveralls.io/repos/github/stephanemery/WineQualityPrediction/badge.svg
    :target: https://coveralls.io/github/stephanemery/WineQualityPrediction
.. image:: https://img.shields.io/badge/docs-latest-orange.svg
    :target: https://github.com/stephanemery/WineQualityPrediction/index.html
.. image:: https://img.shields.io/badge/github-project-0000c0.svg
    :target: https://github.com/stephanemery/WineQualityPrediction
.. image:: https://img.shields.io/github/contributors/badges/shields
    :target: https://github.com/stephanemery/WineQualityPrediction/graphs/contributors
.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :target: https://github.com/stephanemery/WineQualityPrediction/blob/master/LICENSE

  
Wine quality prediction from its physicochemical properties
===========================================================

In this project, we try to predict the wine quality from its physicochemical properties.

Methods used :

1. Preprocessing
    * Drop NaN values
    * Outliers removal
    * Normalize data (Standard or MinMax scaler)
  
2. Models
    * Multi-Linear Regression
    * SVM Regressor
    * KNN Regressor


1 - Clone this repo
-------------------

.. code:: sh

    $ git clone git@https://github.com/stephanemery/WineQualityPrediction WineQualityPrediction
    $ cd WineQualityPrediction


2 - Environment
---------------

This code runs on Python 3.7. To set up the environment, create a new one and install the requirement via pip.

.. code:: sh

    $ conda create -n WineQualityPrediction python=3.7
    $ conda activate WineQualityPrediction
    (WineQualityPrediction) $ pip install requirements.txt


3 - Dataset
-----------

The dataset used are in the folder data_. If the files are not in the folder, they will be downloaded.

.. _data: https://github.com/stephanemery/WineQualityPrediction/tree/master/data


4 - Run
-------

Launch the code by running main.py.

.. code:: sh

   (WineQualityPrediction) $  python main.py
