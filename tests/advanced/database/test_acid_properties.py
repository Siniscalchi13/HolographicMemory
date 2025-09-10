from __future__ import annotations

import sqlite3
import threading
import pytest


@pytest.mark.database
def test_isolation_levels(tmp_path):
    db = tmp_path / "acid.db"
    con1 = sqlite3.connect(db, isolation_level="DEFERRED")
    con2 = sqlite3.connect(db, isolation_level="DEFERRED")
    con1.execute("CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY, v TEXT)")
    con1.commit()

    # Writer starts a transaction
    con1.execute("BEGIN")
    con1.execute("INSERT INTO t (v) VALUES ('x')")

    # Reader in separate connection shouldn't see uncommitted
    count = con2.execute("SELECT COUNT(*) FROM t").fetchone()[0]
    assert count == 0

    # Commit and now visible
    con1.commit()
    count2 = con2.execute("SELECT COUNT(*) FROM t").fetchone()[0]
    assert count2 == 1
    con1.close(); con2.close()

