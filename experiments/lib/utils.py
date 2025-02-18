import black
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import signal
from typing import Generator


def black_print(object: object) -> None:
    print(black.format_str(str(object), mode=black.Mode()))


@contextmanager
def timeout(seconds: int = 1) -> Generator[None, None, None]:
    def timeout_handler(signum: object, frame: object) -> None:
        raise TimeoutError()

    original_handler = signal.signal(signal.SIGALRM, timeout_handler)
    try:
        signal.alarm(seconds)
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)


def setup_shm_symlink(relative_path: str) -> bool:
    """
    Given a relative path, attempts to create a corresponding path in /dev/shm and symlink to it.
    Example: './models/rl1' becomes '/dev/shm/models/rl1'

    Args:
        relative_path: Relative path to the desired output directory

    Returns:
        bool: True if the symlink was created successfully, False if /dev/shm does not exist
    """
    # Check if /dev/shm exists
    if not Path("/dev/shm").exists():
        return False

    # Convert to absolute paths and resolve any .. or . components
    output_dir = Path(relative_path).absolute()
    # Create corresponding path in /dev/shm by using the relative parts
    shm_dir = Path("/dev/shm").joinpath(*Path(relative_path).parts)

    # Ensure parent directories exist
    os.makedirs(output_dir.parent, exist_ok=True)
    os.makedirs(shm_dir.parent, exist_ok=True)

    # Clean up existing output_dir if it exists
    if output_dir.exists() or output_dir.is_symlink():
        if output_dir.is_symlink():
            output_dir.unlink()
        else:
            shutil.rmtree(output_dir)

    # Clean up existing shm_dir if it exists
    if shm_dir.exists():
        shutil.rmtree(shm_dir)

    # Create fresh shm directory
    os.makedirs(shm_dir)

    # Create the symlink
    os.symlink(shm_dir, output_dir)
    return True
