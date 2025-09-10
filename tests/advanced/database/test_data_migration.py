from __future__ import annotations

import sqlite3
import pytest


@pytest.mark.database
def test_simple_schema_migration(tmp_path):
    db = tmp_path / "migrate.db"
    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    cur.execute("INSERT INTO users (name) VALUES ('alice'),('bob')")
    con.commit()

    # Add a new column with default
    cur.execute("ALTER TABLE users ADD COLUMN email TEXT DEFAULT ''")
    con.commit()
    rows = list(cur.execute("SELECT id, name, email FROM users ORDER BY id"))
    assert rows[0][2] == '' and rows[1][2] == ''
    con.close()

