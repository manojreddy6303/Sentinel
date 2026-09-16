"""
backend/tests/test_production_db_safety.py
Tests the production database safety guard to ensure that any attempt by a test
to bind or write to storage/sentinel.db raises an immediate RuntimeError.
"""
import pytest
from database.session import assert_production_db_safety, is_test_environment

def test_is_test_environment_active():
    """Verify that is_test_environment detects test execution."""
    assert is_test_environment() is True

def test_production_db_safety_guard_blocks_production_db():
    """Verify that assert_production_db_safety raises RuntimeError for storage/sentinel.db."""
    with pytest.raises(RuntimeError) as exc_info:
        assert_production_db_safety("sqlite:///C:/Sentinel/storage/sentinel.db")
    assert "PRODUCTION DATABASE PROTECTION ACTIVATED" in str(exc_info.value)

    with pytest.raises(RuntimeError) as exc_info2:
        assert_production_db_safety("sqlite:///storage/sentinel.db")
    assert "PRODUCTION DATABASE PROTECTION ACTIVATED" in str(exc_info2.value)

def test_production_db_safety_guard_allows_test_db():
    """Verify that assert_production_db_safety allows test_sentinel.db or in-memory DB."""
    # Should not raise
    assert_production_db_safety("sqlite:///C:/Sentinel/storage/test_sentinel.db")
    assert_production_db_safety("sqlite:///:memory:")
