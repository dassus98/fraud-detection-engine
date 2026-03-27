import os
import time
import logging
from typing import Optional

import redis

logger = logging.getLogger(__name__)

# Redis hash field names — kept as module-level constants so every method
# uses the same strings and there is no risk of silent key mismatches.
_CARD_KEY_PREFIX = "card:"
_FIELD_TXN_COUNT  = "txn_count"
_FIELD_AMOUNT_SUM = "amount_sum"
_FIELD_LAST_TIME  = "last_txn_time"


class RedisClient:
    """
    Redis-backed real-time feature store for per-card transaction aggregates.

    Each card is stored as a single Redis hash keyed by ``card:{card_id}``.
    Three fields are maintained:

    * ``txn_count``    – number of transactions seen for this card.
    * ``amount_sum``   – running sum of transaction amounts (used to derive mean).
    * ``last_txn_time``– Unix timestamp of the most recent transaction.

    All public methods degrade gracefully: if Redis is unavailable they return
    an empty dict / False rather than raising, so the prediction path is never
    blocked by a cache failure.
    """

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None):
        self.host = host or os.getenv("REDIS_HOST", "localhost")
        self.port = port or int(os.getenv("REDIS_PORT", 6379))
        self.client: Optional[redis.Redis] = None
        self._connect()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        """
        Attempt to open a connection and ping the server.
        Sets ``self.client`` to None on any failure so callers can check
        ``self.available`` instead of catching exceptions everywhere.
        """
        try:
            client = redis.Redis(
                host=self.host,
                port=self.port,
                decode_responses=True,   # all values come back as str, not bytes
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            client.ping()
            self.client = client
            logger.info("Redis connected at %s:%s", self.host, self.port)
        except (redis.ConnectionError, redis.TimeoutError) as exc:
            logger.warning(
                "Redis unavailable (%s:%s) — feature store will return empty dicts. Error: %s",
                self.host, self.port, exc,
            )
            self.client = None

    @property
    def available(self) -> bool:
        """True if a live Redis connection exists."""
        return self.client is not None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _card_key(card_id) -> str:
        return f"{_CARD_KEY_PREFIX}{card_id}"

    # ------------------------------------------------------------------
    # Write methods
    # ------------------------------------------------------------------

    def update_card_features(
        self,
        card_id,
        txn_amount: float,
        txn_time: Optional[float] = None,
    ) -> bool:
        """
        Increment per-card aggregates for a new transaction.

        Uses a Redis pipeline so all three field updates are sent in a
        single round-trip and applied atomically by the server.

        Args:
            card_id:    Any hashable card identifier (int or str).
            txn_amount: Transaction amount in the original currency.
            txn_time:   Unix timestamp; defaults to ``time.time()`` if omitted.

        Returns:
            True on success, False if Redis is unavailable or the write fails.
        """
        if not self.available:
            return False

        key = self._card_key(card_id)
        ts  = txn_time if txn_time is not None else time.time()

        try:
            pipe = self.client.pipeline()
            pipe.hincrbyfloat(key, _FIELD_TXN_COUNT,  1)
            pipe.hincrbyfloat(key, _FIELD_AMOUNT_SUM, txn_amount)
            pipe.hset(key, _FIELD_LAST_TIME, ts)
            pipe.execute()
            return True
        except redis.RedisError as exc:
            logger.warning("Redis write failed for card %s: %s", card_id, exc)
            return False

    def set_card_features(
        self,
        card_id,
        txn_count: int,
        mean_amount: float,
        last_txn_time: float,
    ) -> bool:
        """
        Directly set all aggregates for a card.

        Intended for bulk backfill from the batch pipeline (e.g. pre-loading
        yesterday's counts before the API starts). The amount sum is back-
        computed from count x mean so ``get_card_features`` can derive the
        mean without storing a separate field.

        Returns:
            True on success, False if Redis is unavailable or the write fails.
        """
        if not self.available:
            return False

        key = self._card_key(card_id)
        amount_sum = mean_amount * txn_count

        try:
            self.client.hset(
                key,
                mapping={
                    _FIELD_TXN_COUNT:  txn_count,
                    _FIELD_AMOUNT_SUM: amount_sum,
                    _FIELD_LAST_TIME:  last_txn_time,
                },
            )
            return True
        except redis.RedisError as exc:
            logger.warning("Redis set failed for card %s: %s", card_id, exc)
            return False

    # ------------------------------------------------------------------
    # Read method
    # ------------------------------------------------------------------

    def get_card_features(self, card_id) -> dict:
        """
        Return per-card aggregate features as a flat dict ready to merge
        into a prediction DataFrame row.

        The returned keys are prefixed with ``card_`` to avoid collisions
        with existing model features:

        * ``card_txn_count``    – total transactions seen for this card.
        * ``card_mean_amount``  – mean transaction amount.
        * ``card_last_txn_time``– Unix timestamp of the last transaction.

        Returns an empty dict when Redis is unavailable, when no data has
        been stored for this card yet, or when stored data is malformed.
        Callers should treat an empty dict as no enrichment available
        and proceed with the base features only.
        """
        if not self.available:
            return {}

        key = self._card_key(card_id)

        try:
            raw = self.client.hgetall(key)
        except redis.RedisError as exc:
            logger.warning("Redis read failed for card %s: %s", card_id, exc)
            return {}

        if not raw:
            return {}

        try:
            count      = float(raw[_FIELD_TXN_COUNT])
            amount_sum = float(raw[_FIELD_AMOUNT_SUM])
            last_time  = float(raw[_FIELD_LAST_TIME])
            mean_amount = amount_sum / count if count > 0 else 0.0
        except (KeyError, ValueError, ZeroDivisionError) as exc:
            logger.warning("Malformed Redis data for card %s: %s", card_id, exc)
            return {}

        return {
            "card_txn_count":     count,
            "card_mean_amount":   mean_amount,
            "card_last_txn_time": last_time,
        }
