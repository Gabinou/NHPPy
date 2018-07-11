"""Moran process tests."""
# flake8: noqa
import pytest

from nhppy.discrete import MoranProcess


def test_moran_process_str_repr(n_max):
    instance = MoranProcess(n_max)
    assert isinstance(repr(instance), str)
    assert isinstance(str(instance), str)

def test_moran_process_sample(n_max, n, start):
    instance = MoranProcess(n_max)
    s = instance.sample(n, start)
    assert len(s) <= n
    states = list(range(n_max + 1))
    for state in s:
        assert state in states

def test_moran_process_probability(n_max_fixture):
    with pytest.raises((ValueError, TypeError)):
        instance = MoranProcess(n_max_fixture)

def test_moran_process_n(n, n_fixture):
    instance = MoranProcess(20)
    with pytest.raises((ValueError, TypeError)):
        s = instance.sample(n_fixture, 5)

def test_moran_process_(n, start_fixture):
    instance = MoranProcess(20)
    with pytest.raises((ValueError, TypeError)):
        s = instance.sample(20, start_fixture)
