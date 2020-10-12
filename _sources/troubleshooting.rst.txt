.. vim: set fileencoding=utf-8 :

.. _wine_quality_troubleshooting:

Troubleshooting
===============

You can run unit tests prepared like this (install nose_ first):

.. _nose: https://pypi.org/project/nose/

.. code-block:: shell

  # use your package manager to install the package "nose"
  # here, I examplify with "miniconda":
  (WineQualityPred) $ conda install nose
  (WineQualityPred) $ nosetests test.py
  ....
  ----------------------------------------------------------------------
  Ran 19 tests in 5.897s

  OK

In case of problems, please get in touch with me `by e-mail
<mailto:john.doe@example.com>`_.
