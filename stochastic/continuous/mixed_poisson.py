"""Mixed poisson processes."""

from stochastic.base import Checks
from stochastic.continuous import PoissonProcess

class MixedPoissonProcess(PoissonProcess):
    r"""Mixed poisson process.

    A mixed poisson process is a Poisson process for which the rate is a random variate, 
    a sample taken from a random distribution. On every call of the sample, a new random rate is generated.
    A Poisson process with rate :math:`\lambda` is a count of occurrences of
    i.i.d. exponential random variables with mean :math:`1/\lambda`. This class
    generates samples of times for which cumulative exponential random variables occur. 

    :param function rate_func: random distribution of the rate :math:`\lambda`
    :param rate_args: arguments to input into the rate_func function
    :param rate_kwargs: keyword arguments to input into the rate_func function
    """

    def __init__(self, rate_func,*rate_args,**rate_kwargs):
        self.rate_func = rate_func
        self.rate_args = rate_args
        self.rate_kwargs = rate_kwargs
        self.gen_rate()

    def __str__(self):
        return "Mixed Poisson process with rate {r}.".format(r=str(self.rate))

    def __repr__(self):
        return "MixedPoissonProcess(rate={r})".format(r=str(self.rate))

    @property
    def rate_func(self):
        """Current rate's random distribution."""
        return self._rate_func
        
    @rate_func.setter
    def rate_func(self, value):
        self._rate_func = value
        self.gen_rate()
        
    @property
    def rate_kwargs(self):
        """Parameters for rate generation using given random distribution."""
        return self._rate_kwargs
        
    @rate_kwargs.setter
    def rate_kwargs(self, value):
        self._rate_kwargs = value
        self.gen_rate()
        
    @property
    def rate_args(self):
        """Parameters for rate generation using given random distribution."""
        return self._rate_args
        
    @rate_args.setter
    def rate_args(self, value):
        self._rate_args = value
        self.gen_rate()
        
    @property
    def rate(self):
        """Current rate."""
        return self._rate
        
    @rate.setter
    def rate(self, value):
        rate_func, rate_args,rate_kwargs = value 
        self._rate_func=rate_func
        self._rate_args=rate_args
        self._rate_kwargs=rate_kwargs
        self.gen_rate()
        
    def gen_rate(self):
        if (hasattr(self,'_rate_args')) & (hasattr(self,'_rate_func')) & (hasattr(self,'_rate_kwargs')) : 
            self._rate = self._rate_func(*self._rate_args,**self.rate_kwargs)
            self._check_nonnegative_number(self._rate, "Arrival rate")

    def sample(self, n=None, length=None, zero=True):
        out=super().sample(n, length, zero)
        self.gen_rate()
        """Generate a new random rate upon each realization."""
        return out