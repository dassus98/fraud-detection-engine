import redis
import os
import logging

class RedisClient:
    """
    Docstring for RedisClient
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