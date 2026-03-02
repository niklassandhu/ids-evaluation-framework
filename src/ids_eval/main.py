from __future__ import annotations

import logging.config

from ids_eval.config.logging_config import LOGGING_CONFIG

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("ids_eval")


def main() -> None:
    try:
        from ids_eval.cli import app

        app()
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        exit(1)


if __name__ == "__main__":
    main()
