"""ftplatform/db.py::backup() -- a consistent snapshot of the platform database."""
from __future__ import annotations

import sqlite3

from ftplatform.customers import store
from ftplatform.db import backup, connect


def test_backup_creates_a_readable_copy(tmp_path):
    conn = connect(tmp_path / "customers.db")
    store.create(conn, "acme", "Acme Inc", "saas_support")

    dest = backup(conn, tmp_path / "backups" / "snapshot.db")

    assert dest.exists()
    copy = sqlite3.connect(dest)
    row = copy.execute("SELECT id, name FROM customers WHERE id = 'acme'").fetchone()
    copy.close()
    assert row == ("acme", "Acme Inc")


def test_backup_creates_missing_parent_directories(tmp_path):
    conn = connect(tmp_path / "customers.db")
    dest = backup(conn, tmp_path / "nested" / "deeper" / "snapshot.db")
    assert dest.exists()


def test_backup_is_independent_of_the_live_connection(tmp_path):
    # The snapshot must not change after later writes to the live db --
    # otherwise it isn't a backup, it's a second handle on the same file.
    conn = connect(tmp_path / "customers.db")
    store.create(conn, "acme", "Acme Inc", "saas_support")
    dest = backup(conn, tmp_path / "snapshot.db")

    store.create(conn, "globex", "Globex Corp", "saas_support")

    copy = sqlite3.connect(dest)
    count = copy.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
    copy.close()
    assert count == 1  # globex was added after the snapshot was taken
