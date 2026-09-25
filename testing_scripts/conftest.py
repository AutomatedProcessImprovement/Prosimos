import shutil
from pathlib import Path

import pytest

ASSETS = Path(__file__).resolve().parent / "assets"
LARGE_FILE_BYTES = 1_000_000


@pytest.fixture
def private_assets(tmp_path):
    """Returns a function giving a private copy of testing_scripts/assets, or of one of its
    sub folders, for a test to read, edit and write outputs into. Files are copied; nested
    folders and large data files are linked instead, since tests only read those, and the
    check below still catches a test writing through a link."""

    def copy(sub_folder=None):
        source = ASSETS / sub_folder if sub_folder else ASSETS
        target = tmp_path / "assets" / (sub_folder or "")
        target.mkdir(parents=True)
        for item in source.iterdir():
            if item.is_dir() or item.stat().st_size > LARGE_FILE_BYTES:
                (target / item.name).symlink_to(item)
            else:
                shutil.copy2(item, target / item.name)
        return target

    return copy


def _snapshot(folder):
    return {path: (stat.st_size, stat.st_mtime_ns)
            for path in folder.rglob("*") if path.is_file() and not path.name.startswith(".")
            for stat in [path.stat()]}


@pytest.fixture(autouse=True)
def assets_folder_left_unchanged():
    """Fail any test that writes into the shared assets folder. Tests run one after another
    against the same files, so a test that edits a config or leaves output behind silently
    changes what later tests see. Write modified configs and outputs to tmp_path instead."""
    before = _snapshot(ASSETS)
    yield
    after = _snapshot(ASSETS)

    changed = sorted(str(p.relative_to(ASSETS)) for p in before.keys() & after.keys() if before[p] != after[p])
    created = sorted(str(p.relative_to(ASSETS)) for p in after.keys() - before.keys())
    deleted = sorted(str(p.relative_to(ASSETS)) for p in before.keys() - after.keys())
    problems = [f"{label}: {files}" for label, files in
                (("changed", changed), ("created", created), ("deleted", deleted)) if files]
    if problems:
        pytest.fail("test wrote into testing_scripts/assets; use tmp_path instead. " + "; ".join(problems))
