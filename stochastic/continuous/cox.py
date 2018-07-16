"""Cox processes. Doubly stochastic poisson processes."""
import inspect
import sys
import numpy as np, scipy.interpolate

from stochastic.base import Checks

class CoxProcess(Checks):
    r"""Cox process, or doubly stochastic Poisson process (DSPP).

    # .. image:: _static/poisson_process.png
        # :scale: 50%
    A Poisson process whose rate function is another stochastic process. Lambdaa has to be a NHPPy/stochastic class, it automatically creates a new lambdaa on every sample generation. Otherwise, the NHPP class can be used to generate a Cox process if realizations are given to an NHPP instance as the lambda parameter.

    :param class infoprocess: class from the NHPPy/stochastic that enables the sampling of a stochastic process. Normally, one that outputs
    :param list infoparams: Parameters to input into the information process
    :param float infolength: Length for which to generate a realization of the information process.
    """

    def __init__(self,infoprocess,infoparams,infolength):
        self.infoprocess=infoprocess
        self.infoparams=infoparams
        self.infolength=infolength
        self._check_child(self.infoprocess)
        self.lambdaa= (infoprocess,infoparams,infolength)
        
    @property
    def infolength(self):
        """Current rate's random distribution."""
        return self._infolength
        
    @infolength.setter
    def infolength(self,value):
        """Current rate's random distribution."""
        self._infolength=value     
        
    @property
    def infoprocess(self):
        """Current rate's random distribution."""
        return self._infoprocess
        
    @infoprocess.setter
    def infoprocess(self, value):
        self._infoprocess = value
        if (hasattr(self,'_infoprocess')) & (hasattr(self,'_infoparams')) : 
            infoinstance=self._infoprocess(self._infoparams)
            self.lambdaa =infoinstance.sample(n)

    @property
    def infoparams(self):
        """Parameters for rate generation using given random distribution."""
        return self._ratedistparams
        
    @infoparams.setter
    def infoparams(self, value):
        self._infoparams = value
        if (hasattr(self,'_infoprocess')) & (hasattr(self,'_infoparams')) : 
            if inspect.isclass(self._infoprocess):
                self._infoprocess=self._infoprocess(self._infoparams)
            else:
                self.lambdaa =self._infoprocess(self._infoparams).sample(n)
        
    @property
    def lambdaa(self):
        """Current stochastically generated rate over time."""
        return self._lambdaa
        
    @lambdaa.setter
    def lambdaa(self, value):
        infoprocess, infoparams, infolength = value 
        self._infoprocess= infoprocess
        infoinstance=self._infoprocess(self._infoparams)
        self._lambdaa=infoinstance.sample(length=self._infolength)
    
    def genrate(self):
        infoinstance=self._infoprocess(self._infoparams)
        self._lambdaa=infoinstance.sample(length=self._infolength)

    def _sample_poisson_process(self, n=None,length=None, blocksize=1000):
        """Generate a realization of a Non-homogeneous Poisson process using the Thinning/acceptance-rejection algorithm.

        Stops the generation either if n samples are generated in the length interval, or if a sample is generated outside the length interval.
        """
        if self.lambdaa[-1]>length:
            Time=np.arange(0,length,length/1000)
        else:
            Time=np.arange(0,self.lambdaa[-1],self.lambdaa[-1]/1000)
        Conditionn,Conditionlength=0,0
        flambdaa=scipy.interpolate.interp1d(self.lambdaa,np.cumsum(self.lambdaa>0),kind='cubic')
        lambdamax=np.amax(flambdaa(Time))
        AllDeltaT=[]
        while (Conditionn | Conditionlength)==0:
            U1=np.random.uniform(size=blocksize)
            deltaT=-np.log(U1)/lambdamax
            AllDeltaT=np.append(AllDeltaT,deltaT)
            UThinned=np.cumsum(AllDeltaT)
            UnThinned=UThinned[UThinned<Time[-1]]
            Criteria=flambdaa(UnThinned)/lambdamax
            Conditionlength+=UThinned[-1]>Time[-1]  
            U2=np.random.uniform(size=len(UnThinned))
            TooManyThinned=UnThinned[U2<Criteria]
            Conditionn+=len(TooManyThinned)>n
            Cumul=TooManyThinned[:n]
            if (Conditionlength) & (len(Cumul)<n):
                Cumul=np.append(Cumul,np.zeros(n-len(Cumul)))
        return(Cumul)
    
    def sample(self, n=None,length=None):
        """Generate a realization.

        Exactly one of the following parameters must be provided.

        :param int n: the number of arrivals to simulate
        :param int length: the length of time to simulate; will generate
            arrivals until length is met or exceeded.
        :param bool zero: if True, include :math:`t=0`
        """
        

        out= self._sample_poisson_process(n,length)
        """Generate a new random information process realization upon each Cox realization."""
        self.genrate()
        return out

    def times(self, *args, **kwargs):
        """Disallow times for this process."""
        raise AttributeError("CoxProcess object has no attribute times.")
# logic to checking if the passed instance to the cox process is legit: Checks OR continuous are parents of all process. Checks is parents of all processes. So just checking isinstance of checks SHOULD be enough.
# hat to do about instances of non continuous processes
class MyClass:
    """A simple example class"""
    
    i = 12345
    def __init__(self,a):
        return 
    def f(self):
        return 'hello world'
from poisson import PoissonProcess
from stochastic.base import Continuous
A=CoxProcess(PoissonProcess,1,100)
# print(A.lambdaa)
# print(A.genrate())
# print(A.lambdaa)
print(A.sample(n=10,length=1000))
sys.exit()
# A=PoissonProcess(2)
print(isinstance(A,Continuous))
print(isinstance(A,Checks))
print(isinstance(A,Checks))
print(issubclass(A.__class__,Continuous))
print(issubclass(A.__class__,Checks))
print(issubclass(PoissonProcess,Continuous))
print(issubclass(PoissonProcess,Checks))
print(issubclass(Continuous,Checks))
print(isinstance(A,PoissonProcess))
sys.exit()
