from __future__ import annotations

import sqlite3
import pytest


@pytest.mark.database
def test_rollback_on_error(tmp_path):
    con = sqlite3.connect(tmp_path / "roll.db")
    cur = con.cursor()
    cur.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT UNIQUE)")
    con.commit()

    con.execute("BEGIN")
    cur.execute("INSERT INTO t (v) VALUES ('a')")
    with pytest.raises(sqlite3.IntegrityError):
        cur.execute("INSERT INTO t (v) VALUES ('a')")
        con.commit()
    # Should auto-rollback on exception in sqlite
    con.rollback()
    assert cur.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 1
    con.close()

