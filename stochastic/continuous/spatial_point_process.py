"""Spatial Point Processes."""

from stochastic.continuous.poisson import PoissonProcess
import inspect
import numpy as np

class SpatialPointProcess(PoissonProcess):
    r"""Spatial Point Process.

    .. image:: _static/mixed_poisson_process.png
        :scale: 50%

    A mixed poisson process is a Poisson process for which the rate is
    a scalar random variate. The sample method will generate a random variate
    for the rate before generating a Poisson process realization with the rate.
    A Poisson process with rate :math:`\lambda`
    is a count of occurrences of i.i.d. exponential random
    variables with mean :math:`1/\lambda`. Use the ``rate`` attribute to get
    the most recently generated random rate.

    :param callable density: either a callable or n-dimensional array
    :param tuple bounds: only positinal argument of ``density`` if it is
    callable
    :param dict rate_kwargs: keyword args for callable ``density``
    """

    def __init__(self, density, bounds=(), density_kwargs={}):
        self.density = density
        self.bounds = bounds
        self.density_kwargs = density_kwargs

    def __str__(self):
        return "Spatial point process with specified density function in multidimensional space."

    def __repr__(self):
        return "MixedPoissonProcess(" \
            "density={rf}, bounds={ra}, rate_kwargs={rkw})".format(
                rf=str(self.density),
                ra=str(self.bounds),
                rkw=str(self.rate_kwargs)
            )

    @property
    def density(self):
        """Probability density in the data space."""
        return self._density

    @density.setter
    def density(self, value):
        if (not callable(value)) & (not isinstance(value, (list, tuple, np.ndarray))):
            raise ValueError("Density must be a callable or n-dimensional array.")
        self._density = value

    @property
    def bounds(self):
        """Positional arguments for the density function. 
        Delimits the point generation space. """
        return self._bounds

    @bounds.setter
    def bounds(self, value):
        if not isinstance(value, (list, tuple, np.ndarray)):
            raise ValueError("Bounds must be a list, tuple or np.ndarray.")
        self._bounds = value

    @property
    def rate_kwargs(self):
        """Keyword arguments for the density function."""
        return self._rate_kwargs

    @rate_kwargs.setter
    def rate_kwargs(self, value):
        if not isinstance(value, dict):
            raise ValueError("Rate kwargs must be a dict.")
        self._rate_kwargs = value

    def sample(self, n=None, length=None, zero=True):
        """Generate a realization.

        Exactly one of `n` and `length` must be provided. Generates a random
        variate for the rate, then generates a Poisson process realization
        using this rate.

        :param int n: the number of arrivals to simulate
        :param int length: the length of time to simulate; will generate
            arrivals until length is met or exceeded.
        :param bool zero: if True, include :math:`t=0`
        """
        self.rate = self._sample_rate()
        return self._sample_poisson_process(n, length, zero)
