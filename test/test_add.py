"""
A simple test to ensure Python is being built correctly
"""

import tblis
import pytest

@pytest.mark.parametrize("a,b,expected", [
    (5,  5, 10),
    (0,  0,  0),
    (-5, 5,  0),
])
def test_add(a, b, expected):
    assert tblis.add(a, b) == expected
