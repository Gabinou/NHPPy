"""Spatial Point Processes."""

from stochastic.continuous.poisson import PoissonProcess
import numpy as np
import scipy


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

    :param callable density: either a callable or n-dimensional array. Note:
    Only a single positional argument is assumed, a n-dimensinoal array.
    :param tuple bounds: Bounds in wihch to simulate samples. The number of
    bound pairs must be the same as input arguments of callable density or
    number of the density array.
    :param dict density_kwargs: keyword args for callable ``density``
    """

    def __init__(self, density, density_kwargs={}):
        self.density = density
        self.density_kwargs = density_kwargs

    def __str__(self):
        return "Spatial point process with specified density function\
                in multidimensional space."

    def __repr__(self):
        return "MixedPoissonProcess(" \
            "density={d}, density_kwargs={dkw})".format(
                d=str(self.density),
                dkw=str(self.density_kwargs))

    @property
    def density(self):
        """Probability density in the data space."""
        return self._density

    @density.setter
    def density(self, value):
        if ((not callable(value)) &
           (not isinstance(value, (list, tuple, np.ndarray)))):
            raise ValueError("Density must be a callable or" 
                             "n-dimensional array, list or tuple.")
        self._density = value

    @property
    def density_kwargs(self):
        """Keyword arguments for the density function."""
        return self._density_kwargs

    @density_kwargs.setter
    def density_kwargs(self, value):
        if not isinstance(value, dict):
            raise ValueError("Density kwargs must be a dict.")
        self._density_kwargs = value

    def _sample_spatial_point_process(self, n=None, bounds=(),
                                      algo='thinning', blocksize=1000):
        if (n is not None) & (bounds is not ()) & (bounds is not None):
            Thinned = np.empty((1, len(bounds)))
            if algo == 'thinning':
                if callable(self.density):
                    try:
                        self.density(range(len(bounds)), **self.density_kwargs)
                    except IndexError:
                        raise ValueError("Number of dimensions of bounds must\
                                          match density function")
                    boundstuple = []
                    for i in bounds: boundstuple += (tuple(i),)
                    max = scipy.optimize.minimize(lambda x: -self.density(x),
                                                  x0=[np.mean(i) for i in bounds],
                                                  bounds=boundstuple)
                    density_max = self.density(max.x, *self.density_kwargs)
                    while len(Thinned) < n:
                        Unthinned = np.empty(blocksize)
                        for bound in bounds:
                            Unthinned = np.vstack((Unthinned,
                                                   np.random.uniform(*bound,
                                                   size=(blocksize)))
                                                  )
                        Unthinned = Unthinned[1:, :]
                        U = np.random.uniform(size=(blocksize))
                        Criteria = self.density(Unthinned)/density_max
                        Thinned = np.vstack((Thinned,
                                             Unthinned[:, U < Criteria].T))
                elif isinstance(self.density, (list, tuple, np.ndarray)):
                    if len(self.density.shape) != len(bounds):
                        raise ValueError("Number of dimensions of bounds must\
                                         match density array")
                    density_max = np.amax(self.density)
                    while len(Thinned) < n:
                        Unthinned = np.empty(blocksize, dtype='int')
                        # int: the outputs are indexes of the density array.
                        for shape in self.density.shape:
                            Unthinned = np.vstack((Unthinned,
                                                  np.random.randint(0,
                                                                    shape,
                                                                    size=(blocksize))))
                        Unthinned = Unthinned[1:, :]
                        U = np.random.uniform(size=(blocksize))
                        Criteria_ndim = self.density/density_max
                        Criteria = []
                        for point in Unthinned.T:
                            Criteria.append(Criteria_ndim[tuple(point)])
                        Thinned = np.vstack((Thinned.astype('int'),
                                             Unthinned[:, U < Criteria].T))
                return(Thinned[1:n+1, :])
        else:
            raise ValueError("Must provide arguments bounds and n.")

    def sample(self, n=None, bounds=(), algo='thinning', blocksize=1000):
        """Generate a realization.

        The number of samples ``n`` and the spatial ``bounds`` must be provided.
        The number of dimensions is taken to be the number of bound pairs in
        ``bounds``

        :param int n: the number of arrivals to simulate
        :param tuple bounds: the bounds in which to generate arrivals
        :param str algo: the general algorithm
        """
        return self._sample_spatial_point_process(n, bounds, algo, blocksize)
