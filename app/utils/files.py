from pathlib import Path


def ensure_directory_exists(path: Path) -> None:
    """
    Ensure that the given directory exists; create it if necessary.

    Args:
        path: Path object of the directory to check or create.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise
