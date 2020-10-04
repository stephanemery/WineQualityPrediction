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
       Score : 0.2527510101619952
   KNN Regressor
       Score : 0.17836835834516326
   SVM Regressor
       Score : 0.2479919930723996

   Score for the white wine :
   Multi-Linear Regression
       Score : 0.17457961500511143
   KNN Regressor
       Score : 0.19431050969654862
   SVM Regressor
       Score : 0.31709267173094546

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