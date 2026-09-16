"""
Script to safely purge synthetic automated test fixtures from storage/sentinel.db.
Preserves all authentic security cases, sample videos, and real intelligence records.
"""
import sqlite3
from pathlib import Path

TEST_CASE_PATTERNS = [
    "%Isolated Case%",
    "%Topology Case%",
    "%Duplicate Link Case%",
    "%Large Timeline Case%",
    "%Completely Empty Case%",
    "Case 320x240%",
    "Case 640x480%",
    "Case 1280x720%",
    "Case 1920x1080%",
    "Case 3840x2160%",
    "%Invalid ID Test Case%",
    "%Lifecycle Audit Case%",
    "%Video Linking Case%",
    "%Camera Linking Case%",
    "%Incident Linking Case%",
    "%Explainability Audit%",
    "%Security Event Explainability%",
    "%Export Verification Case%",
    "%Bookmark Workflow Case%",
    "%Notes Audit Case%",
    "%Overlay Non-Mutation Case%",
    "%Investigation Case Alpha%",
    "%Investigation Case Bravo%",
    "%Note Isolation%",
    "%Annotation Isolation%",
]

def clean_test_fixtures(db_path: str = "storage/sentinel.db"):
    db_file = Path(db_path)
    if not db_file.exists():
        print(f"Database {db_path} not found.")
        return

    conn = sqlite3.connect(str(db_file))
    conn.execute("PRAGMA foreign_keys=ON")
    cursor = conn.cursor()

    total_deleted = 0
    for pattern in TEST_CASE_PATTERNS:
        cursor.execute("SELECT id, case_number, title FROM cases WHERE title LIKE ?", (pattern,))
        matches = cursor.fetchall()
        for cid, cnum, title in matches:
            print(f"Deleting test case: {cnum} - {title}")
            cursor.execute("DELETE FROM cases WHERE id = ?", (cid,))
            total_deleted += 1

    conn.commit()
    conn.close()
    print(f"Successfully cleaned {total_deleted} test fixture cases from {db_path}.")

if __name__ == "__main__":
    clean_test_fixtures()
