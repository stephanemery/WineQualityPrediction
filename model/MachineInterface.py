import numpy

class Machine:

    def __init__(self, theta, norm=None):
        """
        **Parameters:**

         - theta (numpy.ndarray): A 1D array (with a dtype of float64) containing the parameters of the logistic regression to be fitted. This variable corresponds to a vector with two positions with the values of $\theta_0$, $\theta_1$ and so on, in that order.

         - norm (tuple): Normalization parameters, a tuple with two components: the mean and the standard deviation. Each component must be an iterable (or `numpy.ndarray`) with a normalization factor for all terms. For the bias term (entry 0), set the mean to 0.0 and the standard deviation to 1.0.
        """         
        
        # When keeping a numpy.ndarray inside an object, making part of the
        # object's state, it's prudent to get a copy of it, so external callers
        # cannot alter it.
        self.theta = numpy.array(theta).copy()

        if norm is not None:
            self.norm = (
                numpy.array(norm[0]).copy(),
                numpy.array(norm[1]).copy(),
            )

            # check
            if self.norm[0].shape != self.norm[1].shape:
                raise (RuntimeError, "Normalization parameters differ in shape")

            if self.norm[0].shape != self.theta.shape:
                raise (RuntimeError, "Normalization parameters and theta differ in shape")

        else:
            # don't normalize at all
            self.norm = (
                numpy.zeros(self.theta.shape),
                numpy.ones(self.theta.shape),
            )

    def set_norm(self, X):
        """Sets the normalization parameters for this machine from a dataset.

        Parameters:

          X (numpy.ndarray): A 2D array with dtype ``float64`` that contains all
            observations for the input variable. This matrix contains N variables,
            which are derived from the initial 4 variables in the data set applying
            the function ``make_polynom()``, similarly (but more generic) to the
            one you defined on the first lab. Notice that, for this exercise, the
            bias column **is already added by the caller**.

        """

        self.norm = (
            numpy.zeros(self.theta.shape),
            numpy.ones(self.theta.shape),
        )

        # sets all terms, but the bias
        self.norm[0][1:] = numpy.mean(X[:, 1:], axis=0)
        self.norm[1][1:] = numpy.std(X[:, 1:], axis=0, ddof=1)

    def normalize(self, X):
        """Normalizes the given dataset by removing the mean and dividing by the
        standard deviation.

        .. important::

           **You must use ``self.norm`` on your answer**


        Parameters:

          X (numpy.ndarray): A 2D array with dtype ``float64`` that contains all
            observations for the input variable. This matrix contains N variables,
            which are derived from the initial 4 variables in the data set applying
            the function ``make_polynom()``, similarly (but more generic) to the
            one you defined on the first lab. Notice that, for this exercise, the
            bias column **is already added by the caller**.


        Returns:

          numpy.ndarray: A 2D array with the same shape as ``X``, but with all
          elements normalized.

        """
        # TODO: Replace the following line with the correct normalization
       
        #Normalize X
        #print ('Normalizing input data X := (X-mean(X))/std(X)')
        return (X - self.norm[0])/self.norm[1]

    def __call__(self, X, pre_normed=False):
        """Spits out the hypothesis given the data.

        Parameters:

          X (numpy.ndarray): A 2D array with dtype ``float64`` that contains all
            observations for the input variable. This matrix contains N variables,
            which are derived from the initial 4 variables in the data set applying
            the function ``make_polynom()``, similarly (but more generic) to the
            one you defined on the first lab. Notice that, for this exercise, the
            bias column **is already added by the caller**.

          pre_normed (bool): ``True``, indicating the input data ``X`` has
            *already* been normalized and that a second normalization step is
            therefore not necessary. By default, this value is ``False``, meaning
            that objects of this class will auto-normalize the input.


        Returns:

          numpy.ndarray: A 1D array (with dtype ``float64``) with the logistic
          hypothesis, i.e., the estimated value for y (:math:`\hat y`), for every
          row of ``X``. The size of the output should match, therefore, the number
          of rows in ``X``.

        """

        Xnorm = X if pre_normed else self.normalize(X)

        # TODO: Port here, the hypothesis for logistic regression
        return 1. / (1. + numpy.exp(-Xnorm@self.theta))
        

    def J(self, X, y, pre_normed=False):
        """
        Calculates the logistic regression cost


        Parameters:

          X (numpy.ndarray): A 2D array with dtype ``float64`` that contains all
            observations for the input variable. This matrix contains N variables,
            which are derived from the initial 4 variables in the data set applying
            the function ``make_polynom()``, similarly (but more generic) to the
            one you defined on the first lab. Notice that, for this exercise, the
            bias column **is already added by the caller**.

          y (numpy.ndarray): A 1D array with dtype ``float64`` containing all
            expected values (labels) for the output variable (i.e., either ones or
            zeroes)

          pre_normed (bool): ``True``, indicating the input data ``X`` has
            *already* been normalized and that a second normalization step is
            therefore not necessary. By default, this value is ``False``, meaning
            that objects of this class will auto-normalize the input.


        Returns:

          float: A single value containing the average cost (iow. loss, error) for
          the predictions you do for ``y`` (i.e., :math:`\hat y`), for **all**
          observation in ``X``, given your (linear) model represented by
          :math:`\theta`.

        """

        Xnorm = X if pre_normed else self.normalize(X)

        # TODO: Port here, the cost for logistic regression
        #import nbimporter does not work :/
        
        h = self(Xnorm, pre_normed=True)
        logh = numpy.nan_to_num(numpy.log(h))
        log1h = numpy.nan_to_num(numpy.log(1-h))


        return -(y*logh +((1-y)*log1h)).mean()



    def dJ(self, X, y, pre_normed=False):
        """
        Calculates the logistic regression first derivative of the cost


        Parameters:

          X (numpy.ndarray): A 2D array with dtype ``float64`` that contains all
            observations for the input variable. This matrix contains N variables,
            which are derived from the initial 4 variables in the data set applying
            the function ``make_polynom()``, similarly (but more generic) to the
            one you defined on the first lab. Notice that, for this exercise, the
            bias column **is already added by the caller**.

          y (numpy.ndarray): A 1D array with dtype ``float64`` containing all
            expected values (labels) for the output variable (i.e., either ones or
            zeroes)

          pre_normed (bool): ``True``, indicating the input data ``X`` has
            *already* been normalized and that a second normalization step is
            therefore not necessary. By default, this value is ``False``, meaning
            that objects of this class will auto-normalize the input.


        Returns:

          numpy.ndarray: A 1D array (with a dtype of ``float64``) containing the
          average cost derivative w.r.t. each individual parameter :math:`theta`
          for the predictions you do for ``y`` (i.e., :math:`\hat y`), taking into
          consideration **all** observations in ``X`` and given your (logistic
          regression) model represented by :math:`\theta`.

        """

        Xnorm = X if pre_normed else self.normalize(X)


        # TODO: Port here, the derivative of the cost for logistic regression


 
        return ((self(Xnorm, pre_normed=True)-y)* Xnorm.T).T.mean(axis=0)
        

    def CER(self, X, y, pre_normed=False):
        """
        Calculates the (vectorized) classification error rate for a 2-class
        problem.


        Parameters:

          X (numpy.ndarray): A 2D array with dtype ``float64`` that contains all
            observations for the input variable. This matrix contains N variables,
            which are derived from the initial 4 variables in the data set applying
            the function ``make_polynom()``, similarly (but more generic) to the
            one you defined on the first lab. Notice that, for this exercise, the
            bias column **is already added by the caller**.

          y (numpy.ndarray): A 1D array with dtype ``float64`` containing all
            expected values (labels) for the output variable (i.e., either ones or
            zeroes)

          pre_normed (bool): ``True``, indicating the input data ``X`` has
            *already* been normalized and that a second normalization step is
            therefore not necessary. By default, this value is ``False``, meaning
            that objects of this class will auto-normalize the input.


        Returns:

          float: The overall classification error rate. A number between 0.0 and
          1.0.

        """

        Xnorm = X if pre_normed else self.normalize(X)
        # TODO: Port here, the classification error rate from exercise 1
        h_ =self(Xnorm, pre_normed=True)
        h_[h_<0.5] =0.0
        h_[h_>=0.5] =1.0
        errors = (h_!=y).sum()
        return float(errors)/len(X)
       
    def save(self, h5f):
        """Saves the machine to a pre-opened HDF5 file


        Parameters:

          h5f (bob.io.base.HDF5File): An HDF5 file that has been opened for writing
            and pre-set so that this machine dumps its parameters on the expected
            location.

        """

        # We save all parameters that make-up the "state" of this "machine", so we
        # can reload it from file and re-create the existing settings at any time
        h5f.set('theta', self.theta)
        h5f.set('subtract', self.norm[0])
        h5f.set('divide', self.norm[1])

    def load(self, h5f):
        """Loads the machine from a pre-opened HDF5 file


        Parameters:

          h5f (bob.io.base.HDF5File): An HDF5 file that has been opened for reading
            and pre-set so that this machine reads its parameters from the expected
            location.

        """

        # Recover theta, norm[0] and norm[1] from file
        self.theta = h5f.read('theta')
        self.norm = h5f.read('subtract'), h5f.read('divide')
