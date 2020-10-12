.. vim: set fileencoding=utf-8 :

.. _installation:


Installation
************

From pip :
============

Run the command :

.. code:: sh
   
   $ pip install -i https://test.pypi.org/simple/ wineQualityPred

Download the sources
====================

To download a copy of this package, follow these steps :


1 - Clone this repo
-------------------

.. code:: sh

    $ git clone https://github.com/stephanemery/WineQualityPrediction WineQualityPrediction
    $ cd WineQualityPrediction


2 - Environment
---------------

This code runs on Python 3.7. To set up the environment, create a new one and install the requirement via pip.

.. code:: sh

    $ conda create -n WineQualityPrediction python=3.7
    $ conda activate WineQualityPrediction
    (WineQualityPrediction) $ conda install --file requirements.txt


3 - Dataset
-----------

The dataset used are in the folder data_. If the files are not in the folder, they will be downloaded.

.. _data: https://github.com/stephanemery/WineQualityPrediction/tree/Packaging/wineQualityPred/data
