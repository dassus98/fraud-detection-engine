import redis
import os
import logging

class RedisClient:
    """
    Thin wrapper around Redis that establishes a connection at init and logs a warning on failure.
    """
    def __init__(self, host = None, port = None):
        self.host = host
        self.port = port
        self.client = None

        try:
            self.client = redis.Redis()
            self.client.ping()
        except redis.ConnectionError:
            logging.warning('Redis connection has failed.')
            self.client = None