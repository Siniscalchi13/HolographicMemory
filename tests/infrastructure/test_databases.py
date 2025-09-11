from __future__ import annotations

import sqlite3

import pytest


@pytest.mark.integration
def test_inmemory_sqlite_transactions():
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    cur.execute("create table kv (k text primary key, v text)")
    conn.commit()
    # Insert within tx and rollback
    cur.execute("begin")
    cur.execute("insert into kv(k,v) values(?,?)", ("a", "1"))
    conn.rollback()
    cur.execute("select count(*) from kv")
    assert cur.fetchone()[0] == 0
    # Commit path
    cur.execute("begin")
    cur.execute("insert into kv(k,v) values(?,?)", ("a", "1"))
    conn.commit()
    cur.execute("select v from kv where k=?", ("a",))
    assert cur.fetchone()[0] == "1"

