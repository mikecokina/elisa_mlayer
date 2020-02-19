import logging

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(process)d - %(name)s - %(levelname)s: %(message)s")


def getLogger(name):
    return logging.getLogger(name=name)
