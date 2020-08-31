
import numpy

from machine import Machine as MachineBase

class Machine(MachineBase):

    def J(self, X, y, regularization=0.0, pre_normed=False):
        """
        Calculates the regularized logistic regression cost

        The cost with regularization can be expressed as:

        .. math::

           J(\bm{\theta}) = -\frac{1}{n} \sum_{i=1}^n [y_i \log h_{\bm{\theta}}(\mathbf{x}_i) + (1 - y_i) \log (1 - h_{\bm{\theta}}(\mathbf{x}_i))] + \frac{\lambda}{2} \sum_{j=2}^d \bm\theta_d^2


        Parameters:

          X (numpy.ndarray): A 2D array with dtype ``float64`` that contains all
            observations for the input variable. This matrix contains N variables,
            which are derived from the initial 4 variables in the data set applying
            the function ``make_polynom()``, similarly (but more generic) to the
            one you defined on the first lab

          y (numpy.ndarray): A 1D array with dtype ``float64`` containing all
            expected values (labels) for the output variable (i.e., either ones or
            zeroes)

          regularization (float): The regularization parameter :math:`\lambda`
            which should be considered when evaluating the current cost as defined
            above.

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
        #return 0.0
        # TODO: Port here, the cost for logistic regression from exercise 2 and
        # then modify it to take into consideration the extra regularization term
                
        h = self(Xnorm, pre_normed=True)
        logh = numpy.nan_to_num(numpy.log(h))
        log1h = numpy.nan_to_num(numpy.log(1-h))
        regularizaton_term= regularization*(self.theta[1:]**2).sum()/2.0
        main_term=-(y*logh +((1-y)*log1h)).mean()
        return main_term + regularizaton_term

    def dJ(self, X, y, regularization=0.0, pre_normed=False):
        """
        Calculates the regularized cost derivative w.r.t. the parameters

        The regularized cost derivative w.r.t. :math:`\theta` can be expressed as:

        .. math::

           \frac{dJ}{d\bm\theta_t} = \frac{1}{n} \sum_{i=1}^n (h_{\bm{\theta}}(\mathbf{x}_i) - y_i) x_{i,j} + \lambda \theta_t


        Parameters:

          X (numpy.ndarray): A 2D array with dtype ``float64`` that contains all
            observations for the input variable. This matrix contains N variables,
            which are derived from the initial 4 variables in the data set applying
            the function ``make_polynom()``, similarly (but more generic) to the
            one you defined on the first lab

          y (numpy.ndarray): A 1D array with dtype ``float64`` containing all
            expected values (labels) for the output variable (i.e., either ones or
            zeroes)

          regularization (float): The regularization parameter :math:`\lambda`
            which should be considered when evaluating the current cost as defined
            above.

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
        
        # TODO: Port here, the derivative of the cost for logistic regression from
        # exercise 2 and then modify it to take into consideration the extra
        # regularization term
        retval= ((self(Xnorm, pre_normed=True)- y)*Xnorm.T).T.mean(axis=0)
        retval[1:]+=(regularization*self.theta[1:])
        return retval
    
