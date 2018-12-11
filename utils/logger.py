import logging
from utils.envs import logger_file

def setup_logger(name: str, log_path: str, level) -> logging.Logger:
    """Logger utility for logging messages

    Prints logs to screen via ch (channel handler), and saves logs to via fh (file handler)

    Args:
        name: Name of logger
        log_path: Path to write logs to

    Returns:
        Logger with logging to both screen output and log file in log path

    Examples:
        >>> type(setup_logger('test_logger', 'test.log'))
        <class 'logging.Logger'>
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    # create console handler and set level to info
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(logging.INFO)

    # create file handler that logs debug messages
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)

    # add ch to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger

log = setup_logger('Logger', logger_file, logging.DEBUG)