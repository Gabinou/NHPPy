"""Test SpatialPointProcess."""
# flake8: noqa
import pytest
import inspect 
from stochastic.continuous import SpatialPointProcess


def test_poisson_process_str_repr(density, density_kwargs):
    if callable(density):
            with pytest.raises(ValueError):
                instance = SpatialPointProcess(rate_func, rate_args, rate_kwargs)
        elif not isinstance(rate_kwargs, dict):
            with pytest.raises(ValueError):
                instance = SpatialPointProcess(rate_func, rate_args, rate_kwargs)
        elif not callable(rate_func):
            with pytest.raises(ValueError):
                instance = SpatialPointProcess(rate_func, rate_args, rate_kwargs)
        else: 
        instance = SpatialPointProcess(rate)
    assert isinstance(repr(instance), str)
    assert isinstance(str(instance), str)

def test_poisson_process_sample(density, n_fixture, bounds, density_kwargs):
    instance = SpatialPointProcess(rate)
    if n_fixture is None and length is None:
        with pytest.raises(ValueError):
            s = instance.sample(n_fixture, length, zero)
    elif length is not None and n_fixture is None:
        s = instance.sample(n_fixture, length, zero)
        assert s[-1] >= length
    else:  # n_fixture is not None:
        s = instance.sample(n_fixture, length, zero)
        assert len(s) == n_fixture + int(zero)

def test_poisson_process_times(rate, n):
    instance = SpatialPointProcess(rate)
    with pytest.raises(AttributeError):
        times = instance.times(n)
