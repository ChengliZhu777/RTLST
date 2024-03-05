import logging
import argparse


def set_logging(rank=-1):
    """
    %(message)s : output log content
    %(asctime)s : output logging time
    """
    logging.basicConfig(format='%(message)s',
                        level=logging.INFO if rank in [-1, 0] else logging.WARN)
