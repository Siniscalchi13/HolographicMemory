from __future__ import annotations

import sqlite3
import pytest


@pytest.mark.database
def test_commit_and_rollback_isolated(tmp_path):
    db = tmp_path / "test.db"
    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
    con.commit()

    # Insert in a transaction and rollback
    con.execute("BEGIN")
    cur.execute("INSERT INTO t (v) VALUES ('a')")
    con.rollback()
    assert cur.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 0

    # Insert and commit
    con.execute("BEGIN")
    cur.execute("INSERT INTO t (v) VALUES ('b')")
    con.commit()
    assert cur.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 1
    con.close()

