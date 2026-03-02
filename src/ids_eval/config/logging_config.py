from pathlib import Path


def _determine_repo_root() -> Path:
    current_path = Path(__file__).resolve()
    for parent in [current_path.parent] + list(current_path.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    raise RuntimeError("Could not determine repository root directory as no project marker was found.")


REPO_ROOT = _determine_repo_root()
LOG_DIR = REPO_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE_PATH = str(LOG_DIR / "ids_eval.log")
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "colored": {
            "()": "colorlog.ColoredFormatter",
            "format": "%(asctime)s - %(name)s - %(log_color)s%(levelname)s%(reset)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
            "log_colors": {
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "colored",
            "level": "INFO",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "standard",
            "level": "DEBUG",
            "filename": LOG_FILE_PATH,
            "maxBytes": 10485760,  # 10 MB
            "backupCount": 5,
            "encoding": "utf8",
        },
    },
    "loggers": {
        "": {  # Root logger
            "handlers": ["console", "file"],
            "level": "DEBUG",
        },
        "services": {
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": False,  # Don't pass these logs to the root logger
        },
    },
}
