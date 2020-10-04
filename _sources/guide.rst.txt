.. vim: set fileencoding=utf-8 :

.. _wine_quality_guide:


Guide
============

This guide explains how to use this package and obtain results published in our paper. Results can be re-generated automatically by executing the following command:

.. code-block:: sh

   (WineQualityPrediction) python main.py -ns

By executing this command, you should get these results :

.. testcode::

   import main
   main.main(False)

.. testoutput::
   :options: +NORMALIZE_WHITESPACE
   
   Preprocessing done !
   Score for the red wine :
   Multi-Linear Regression
       Score : 0.252751
   KNN Regressor
       Score : 0.178368
   SVM Regressor
       Score : 0.247992

   Score for the white wine :
   Multi-Linear Regression
       Score : 0.174580
   KNN Regressor
       Score : 0.194311
   SVM Regressor
       Score : 0.317093

You can run main.py with differents options to see how the results change : 

.. code-block:: sh

   usage: main.py [-h] [--scaler SCALER] [-nn] [-ns] [-nro]

   Predict wine quality from its physicochemical properties.

   optional arguments:
     -h, --help            show this help message and exit
     --scaler SCALER       The name of the scaler : "StandardScaler", "MinMaxScaler"
     -nn, --not_normalize  Do not normalize data
     -ns, --not_shuffle    Do not shuffle data
     -nro, --not_remove_outliers
                           Do not remove outliers