"""Miller-index parsing."""
import pytest

from tinykit.slabgen import parse_miller_index


@pytest.mark.parametrize("text,expected", [
    ("111", (1, 1, 1)),
    ("201", (2, 0, 1)),
    ("-201", (-2, 0, 1)),
    ("2,0,1", (2, 0, 1)),
    ("-2,0,1", (-2, 0, 1)),
    ("1,-1,0", (1, -1, 0)),
    ("1 1 1", (1, 1, 1)),
])
def test_parse_miller_index(text, expected):
    assert parse_miller_index(text) == expected
