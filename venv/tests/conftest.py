"""Test configuration and fixtures."""

import pytest
from src.logger import setup_logger

@pytest.fixture
def logger():
    """Provide a logger instance for tests."""
    return setup_logger("test")