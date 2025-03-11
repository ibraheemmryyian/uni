"""Tests for FAQ database functionality."""

import pytest
from src.models.faq_database import FAQDatabase
from src.utils.exceptions import FAQDatabaseError

def test_faq_database_initialization():
    db = FAQDatabase()
    assert db is not None

def test_invalid_faq_data():
    db = FAQDatabase()
    with pytest.raises(FAQDatabaseError):
        db.setup({"invalid": None})