# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from alpha_factory_v1.core.archive.db import ArchiveDB
from alpha_factory_v1.core.tools import dgm_import


def test_dgm_import(tmp_path: Path) -> None:
    log_dir = Path("tests/fixtures/dgm_logs")
    db_path = tmp_path / "archive.db"
    count = dgm_import.import_logs(log_dir, db_path=db_path)
    assert count == 80

    db = ArchiveDB(db_path)
    history = list(db.history("h79"))
    assert len(history) == 80
