"""
pytest configuration and test database isolation for Sentinel.
Directs all test executions to storage/test_sentinel.db so that the primary
application database (storage/sentinel.db) is never polluted with test fixtures.
"""
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Set test database path before database.session is imported
_test_db_dir = _project_root / "storage"
_test_db_dir.mkdir(parents=True, exist_ok=True)
_test_db_path = _test_db_dir / "test_sentinel.db"

# Force test database URL and testing flag for test runs unconditionally
os.environ["SENTINEL_TESTING"] = "1"
os.environ["DATABASE_URL"] = f"sqlite:///{_test_db_path}"

# If database.session was already imported, dynamically rebind to the test database
if "database.session" in sys.modules:
    from database.session import rebind_engine
    rebind_engine(f"sqlite:///{_test_db_path}")

# Now import session and init test db
import pytest
from database.session import init_db, engine, Base

@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    """Ensure test database schema is initialized for tests."""
    init_db()
    yield
