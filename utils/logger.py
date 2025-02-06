import logging
import os


def get_logger(file_name: str):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", file_name)
    )
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
