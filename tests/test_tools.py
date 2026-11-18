import pytest

from src.tools import safe_calculate


def test_basic():
    assert safe_calculate("2 + 3 * 4") == 14


def test_power_and_parentheses():
    assert safe_calculate("(2 ^ 3) / 4") == 2


def test_rejects_code():
    with pytest.raises(Exception):
        safe_calculate("__import__('os').system('ls')")
