import numpy
from machine_ex4 import Machine


class Trainer:
  """A class to handle all training aspects for Logistic Regression

  You **don't** have to implement anything here, this class just uses the above
  Machine implementation to handle all training aspects.

  Parameters:

    normalize (bool): True if we should normalize the dataset (after
      polynomial expansion).

    regularization (float): The regularization parameter :math:`\lambda` which
      should be considered when evaluating the current cost as defined above.

  """


  def __init__(self, normalize=False, regularizer=0.0):

    self.normalize = normalize
    self.regularizer = regularizer


  def J(self, theta, machine, X, y):
    """
    Calculates the vectorized cost *J*.


    Parameters:

      theta (numpy.ndarray): A 1D array (with a dtype of float64) containing
        the parameters of the logistic regression to be fitted. This variable
        corresponds to a vector with two positions with the values of
        :math:`\theta_0`, :math:`\theta_1` and so on, in that order.

      machine (Machine): The machine that is currently being trained

      X (numpy.ndarray): A 2D array with dtype ``float64`` that contains all
        observations for the input variable. This matrix contains N variables,
        which are derived from the initial 4 variables in the data set applying
        the function ``make_polynom()``, similarly (but more generic) to the
        one you defined on the first lab

      y (numpy.ndarray): A 1D array with dtype ``float64`` containing all
        expected values (labels) for the output variable (i.e., either ones or
        zeroes)


    Returns:

      float: A single value containing the average cost (iow. loss, error) for
      the predictions you do for ``y`` (i.e., :math:`\hat y`), for **all**
      observation in ``X``, given your (linear) model represented by
      :math:`\theta`.

    """

    machine.theta = theta
    return machine.J(X, y, self.regularizer)


  def dJ(self, theta, machine, X, y):
    """
    Calculates the vectorized partial derivative of the cost *J* w.r.t. to
    **all** :math:`\theta`'s. Use the training dataset.

    Parameters:

      theta (numpy.ndarray): A 1D array (with a dtype of float64) containing
        the parameters of the logistic regression to be fitted. This variable
        corresponds to a vector with two positions with the values of
        :math:`\theta_0`, :math:`\theta_1` and so on, in that order.

      machine (Machine): The machine that is currently being trained

      X (numpy.ndarray): A 2D array with dtype ``float64`` that contains all
        observations for the input variable. This matrix contains N variables,
        which are derived from the initial 4 variables in the data set applying
        the function ``make_polynom()``, similarly (but more generic) to the
        one you defined on the first lab

      y (numpy.ndarray): A 1D array with dtype ``float64`` containing all
        expected values (labels) for the output variable (i.e., either ones or
        zeroes)


    Returns:

      numpy.ndarray: A 1D array (with a dtype of ``float64``) containing the
      average cost derivative w.r.t. each individual parameter :math:`theta`
      for the predictions you do for ``y`` (i.e., :math:`\hat y`), taking into
      consideration **all** observations in ``X`` and given your (logistic
      regression) model represented by :math:`\theta`.

    """

    machine.theta = theta
    return machine.dJ(X, y, self.regularizer)


  def train(self, X, y):
    """
    Optimizes the machine parameters to fit the input data, using
    ``scipy.optimize.fmin_l_bfgs_b``.


    Parameters:

      X (numpy.ndarray): A 2D array with dtype ``float64`` that contains all
        observations for the input variable. This matrix contains N variables,
        which are derived from the initial 4 variables in the data set applying
        the function ``make_polynom()``, similarly (but more generic) to the
        one you defined on the first lab

      y (numpy.ndarray): A 1D array with dtype ``float64`` containing all
        expected values (labels) for the output variable (i.e., either ones or
        zeroes)


    Returns:

      Machine, ``None``: the trained machine, or ``None``, if there was an
      error.

    """

    # prepare the machine
    theta0 = numpy.zeros(X.shape[1])
    machine = Machine(theta0)
    if self.normalize:
      machine.set_norm(X)

    print('Settings:')
    print('  * initial guess = %s' % ([k for k in theta0],))
    print('  * cost (J) = %g' % (machine.J(X, y, self.regularizer),))
    print('  * CER      = %g%%' % (100*machine.CER(X, y),))
    print('Training using scipy.optimize.fmin_l_bfgs_b()...')

    # Fill in the right parameters so that the minimization can take place
    from scipy.optimize import fmin_l_bfgs_b
    theta, cost, d = fmin_l_bfgs_b(
        self.J, # the cost - will be called like self.J(theta, machine, X, y)
        theta0, # our initial estimate (anything will do it is convex!)
        self.dJ, # cost derivative - called like self.dJ(theta, machine, X, y)
        (machine, X, y), # parameters passed to J and dJ besides theta
        )

    if d['warnflag'] == 0:

      print("** LBFGS converged successfuly **")
      machine.theta = theta
      print('Final settings:')
      print('  * theta = %s' % ([k for k in theta],))
      print('  * cost (J) = %g' % (cost,))
      print('  * CER      = %g%%' % (100*machine.CER(X, y),))
      return machine

    else:
      print("LBFGS did **not** converge:")
      if d['warnflag'] == 1:
        print("  Too many function evaluations")
      elif d['warnflag'] == 2:
        print("  %s" % d['task'])
      return None

