"""Record real circuit states for lightweight, seekable epoch replay."""

import json
import sqlite3
from contextlib import closing


def record(directory, frame):
    with closing(sqlite3.connect(directory / "quantum.sqlite")) as connection:
        with connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS frames (epoch INTEGER PRIMARY KEY, state TEXT NOT NULL)"
            )
            connection.execute(
                "INSERT OR REPLACE INTO frames VALUES (?, ?)",
                (frame["epoch"], json.dumps(frame, allow_nan=False)),
            )


def frames_since(directory, after):
    path = directory / "quantum.sqlite"
    if not path.exists():
        return []
    with closing(sqlite3.connect(path)) as connection:
        # The writer may have created the file before committing its first table.
        if not connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='frames'"
        ).fetchone():
            return []
        rows = connection.execute(
            "SELECT state FROM frames WHERE epoch > ? ORDER BY epoch LIMIT 64", (after,)
        )
        return [json.loads(state) for (state,) in rows]
