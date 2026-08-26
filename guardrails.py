import logging
from pathlib import Path


logger = logging.getLogger("agent")


FORBIDDEN_COMMANDS = [
    "rm -rf /", 
    "rm -rf *", 
    "format", 
    "shutdown", 
    "reboot",
    "fdisk"
]

def validate_command(command: str) -> bool:
    """
    Ellenőrzi, hogy a parancs biztonságos-e.
    Visszatér: True, ha biztonságos; False, ha tiltott.
    """
    for forbidden in FORBIDDEN_COMMANDS:
        if forbidden in command:
            return False
    return True


def check_current_directory(startup_directory: Path) -> bool:
    """
    Ellenőrzi, hogy a program a megfelelő könyvtárban fut-e.
    Visszatér: True, ha a könyvtár megfelelő; False, ha nem.
    """
    current_dir = Path.cwd().resolve()
    startup_dir = startup_directory.resolve()
    if current_dir != startup_dir:
        logger.warning(
            "A program nem a startup könyvtárban fut. Startup: %s, Jelenlegi: %s",
            startup_dir,
            current_dir,
        )
        return False
    return True