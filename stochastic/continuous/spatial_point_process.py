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

    def __init__(self, density, density_args={}, density_kwargs={}):
        self.density = density
        self.bounds = bounds
        self.density_kwargs = density_kwargs

    def __str__(self):
        return "Spatial point process with specified density function in multidimensional space."

    def __repr__(self):
        return "MixedPoissonProcess(" \
            "density={d}, density_args={da}, density_kwargs={dkw})".format(
                rf=str(self.density),
                ra=str(self.density_args),
                rkw=str(self.density_akwargs)
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
    def density_args(self):
        """Positional arguments for the density function. 
        Delimits the point generation space. """
        return self._density_args

    @bounds.setter
    def density_args(self, value):
        if not isinstance(value, (list, tuple)):
            raise ValueError("Density args must be a list, tuple.")
        self._bounds = value

    @property
    def density_kwargs(self):
        """Keyword arguments for the density function."""
        return self._density_kwargs

    @rate_kwargs.setter
    def density_kwargs(self, value):
        if not isinstance(value, dict):
            raise ValueError("Density kwargs must be a dict.")
        self._density_kwargs = value

    def sample(self, bounds=(), n=None, algo='thinning'):
        """Generate a realization.

        The number of samples ``n`` and the spatial ``bounds`` must be provided.
        The number of dimensions is taken to be the number of bound pairs in    
        ``bounds``

        :param int n: the number of arrivals to simulate
        :param tuple bounds: the bounds in which to generate arrivals
        """
        if callable(lambdaa):
            boundstuple=[]
            for i in Boundaries: boundstuple+=(tuple(i),)
            max = scipy.optimize.minimize(lambda x: -lambdaa(*x),x0=[np.mean(i) for i in Boundaries],bounds = boundstuple)
            lmax=lambdaa(*max.x)
        else:
            lmax=np.amax(lambdaa)
        Thinned=[]
        while len(Thinned)<Samples:
            for i in Boundaries:
                if 'Unthin' not in locals():
                    Unthin=np.random.uniform(*i,size=(blocksize))
                else:
                    Unthin=np.vstack((Unthin,np.random.uniform(*i,size=(blocksize))))
            Unthin.T
            if len(Unthin.shape)==1:
                Unthin=np.reshape(Unthin,(1,len(Unthin)))
            U=np.random.uniform(size=(blocksize))
            if callable(lambdaa): 
                Criteria=lambdaa(*Unthin)/lmax
            else:
                Criteria2D=lambdaa/lmax
                Indx=(Unthinx*lambdaa.shape[0]).astype(int)
                Indy=(Unthiny*lambdaa.shape[1]).astype(int)
                Criteria=Criteria2D[Indx,Indy]
                Unthin=np.transpose(np.vstack((Unthinx,Unthiny)))
            if Thinned==[]: 
                Thinned=Unthin.T[U<Criteria,:]
            else:
                Thinned=np.vstack((Thinned,Unthin.T[U<Criteria,:]))
            del Unthin
        Thinned=Thinned[:Samples,:]
        return(Thinned)


        self.rate = self._sample_rate()
        return self._sample_poisson_process(n, length, zero)
