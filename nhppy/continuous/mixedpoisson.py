"""Poisson processes."""
import numpy as np

from stochastic.base import Checks

class MixedPoissonProcess(Checks):
    r"""Mixed Poisson process.

    # .. image:: _static/poisson_process.png
        # :scale: 50%
    A mixed poisson process is a Poisson process for which the rate is a random variate, 
    a sample taken from a random distribution. On every call of the sample, a new random rate is generated.
    A Poisson process with rate :math:`\lambda` is a count of occurrences of
    i.i.d. exponential random variables with mean :math:`1/\lambda`. This class
    generates samples of times for which cumulative exponential random     variables occur. 

    :param function RateDist: random distribution of the rate :math:`\lambda` which defines the rate of
        occurrences of the process
    :param list of floats RateDistParams: Parameters to input into the RateDistFunction
    """

    def __init__(self, RateDist,RateDistParams):
        self.RateDist = RateDist
        self.RateDistParams = RateDistParams
        self._rate=RateDist(*RateDistParams)

    def __str__(self):
        return "Mixed Poisson process with rate {r}.".format(r=str(self.rate))

    def __repr__(self):
        return "PoissonProcess(rate={r})".format(r=str(self.rate))

    @property
    def rate(self):
        """Current rate."""
        return self._rate
        """Current rate random distribution and parameters."""
        

    @rate.setter
    def rate(self, RateDist, RateDistParams):
        self._rate = RateDist(*RateDistParams)
        self._check_nonnegative_number(self._rate, "Arrival rate")


    def _sample_poisson_process(self, n=None, length=None, zero=True):
        """Generate a realization of a Mixed Poisson process.

        Generate a poisson process sample up to count of length if time=False,
        otherwise generate a sample up to time t=length if time=True
        """
        
        if n is not None:
            self._check_increments(n)

            exponentials = np.random.exponential(
                scale=1.0 / self.rate, size=n)

            s = np.array([0] + list(np.cumsum(exponentials)))
            if zero:
                return s
            else:
                return s[1:]
        elif length is not None:
            self._check_positive_number(length, "Sample length")

            t = 0
            times = []
            if zero:
                times.append(0)
            exp_rate = 1.0 / self.rate

            while t < length:
                t += np.random.exponential(scale=exp_rate)
                times.append(t)
            return np.array(times)
            
        else:
            raise ValueError(
                "Must provide either argument n or length.")

    def sample(self, n=None, length=None, zero=True):
        """Generate a realization.

        Exactly one of the following parameters must be provided.

        :param int n: the number of arrivals to simulate
        :param int length: the length of time to simulate; will generate
            arrivals until length is met or exceeded.
        :param bool zero: if True, include :math:`t=0`
        """
        return self._sample_poisson_process(n, length, zero)

    def times(self, *args, **kwargs):
        """Disallow times for this process."""
        raise AttributeError("MixedPoissonProcess object has no attribute times.")