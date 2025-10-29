"""
Bandwidth management utilities for transfer queue.
Implements token-bucket rate limiting and concurrency control.
"""
import time
import threading
from typing import Optional, Dict
from dataclasses import dataclass, field
from datetime import datetime
import redis


@dataclass
class TokenBucket:
    """Token bucket for rate limiting."""
    capacity: int  # Maximum tokens (bytes)
    rate: float  # Refill rate (bytes per second)
    tokens: float = field(init=False)
    last_update: float = field(init=False)
    lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self):
        self.tokens = float(self.capacity)
        self.last_update = time.time()

    def refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_update = now

    def consume(self, tokens: int, timeout: Optional[float] = None) -> bool:
        """
        Consume tokens from the bucket.

        Args:
            tokens: Number of tokens to consume
            timeout: Maximum time to wait for tokens (seconds)

        Returns:
            True if tokens were consumed, False if timeout reached
        """
        start_time = time.time()

        while True:
            with self.lock:
                self.refill()

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True

                # Check timeout
                if timeout is not None:
                    elapsed = time.time() - start_time
                    if elapsed >= timeout:
                        return False

            # Sleep briefly before retrying
            time.sleep(0.01)

    def available_tokens(self) -> int:
        """Get number of available tokens."""
        with self.lock:
            self.refill()
            return int(self.tokens)


class BandwidthManager:
    """
    Manages bandwidth allocation across multiple transfers.
    Implements global and per-transfer rate limiting.
    """

    def __init__(self, global_limit_bps: Optional[int] = None, max_concurrent: int = 3):
        """
        Initialize bandwidth manager.

        Args:
            global_limit_bps: Global bandwidth limit in bytes per second (None = unlimited)
            max_concurrent: Maximum concurrent transfers
        """
        self.global_limit_bps = global_limit_bps
        self.max_concurrent = max_concurrent

        # Global token bucket
        if global_limit_bps:
            self.global_bucket = TokenBucket(
                capacity=global_limit_bps * 2,  # 2 second burst capacity
                rate=global_limit_bps
            )
        else:
            self.global_bucket = None

        # Per-transfer buckets
        self.transfer_buckets: Dict[str, TokenBucket] = {}
        self.lock = threading.Lock()

        # Concurrency control
        self.active_transfers = set()
        self.concurrency_lock = threading.Lock()

    def set_transfer_limit(self, transfer_id: str, limit_bps: int):
        """Set bandwidth limit for a specific transfer."""
        with self.lock:
            self.transfer_buckets[transfer_id] = TokenBucket(
                capacity=limit_bps * 2,  # 2 second burst capacity
                rate=limit_bps
            )

    def remove_transfer(self, transfer_id: str):
        """Remove transfer-specific rate limiter."""
        with self.lock:
            if transfer_id in self.transfer_buckets:
                del self.transfer_buckets[transfer_id]

        with self.concurrency_lock:
            if transfer_id in self.active_transfers:
                self.active_transfers.remove(transfer_id)

    def can_start_transfer(self, transfer_id: str) -> bool:
        """Check if a new transfer can start based on concurrency limit."""
        with self.concurrency_lock:
            if len(self.active_transfers) >= self.max_concurrent:
                return False
            self.active_transfers.add(transfer_id)
            return True

    def finish_transfer(self, transfer_id: str):
        """Mark transfer as finished to free concurrency slot."""
        with self.concurrency_lock:
            if transfer_id in self.active_transfers:
                self.active_transfers.remove(transfer_id)
        self.remove_transfer(transfer_id)

    def throttle(self, transfer_id: str, bytes_to_transfer: int, timeout: Optional[float] = None) -> bool:
        """
        Throttle transfer to respect bandwidth limits.

        Args:
            transfer_id: Transfer identifier
            bytes_to_transfer: Number of bytes about to be transferred
            timeout: Maximum time to wait (seconds)

        Returns:
            True if transfer can proceed, False if timeout reached
        """
        # Check global limit
        if self.global_bucket:
            if not self.global_bucket.consume(bytes_to_transfer, timeout):
                return False

        # Check per-transfer limit
        with self.lock:
            if transfer_id in self.transfer_buckets:
                bucket = self.transfer_buckets[transfer_id]
                if not bucket.consume(bytes_to_transfer, timeout):
                    return False

        return True

    def get_stats(self) -> dict:
        """Get current bandwidth manager statistics."""
        with self.concurrency_lock:
            active_count = len(self.active_transfers)

        global_available = None
        if self.global_bucket:
            global_available = self.global_bucket.available_tokens()

        return {
            "active_transfers": active_count,
            "max_concurrent": self.max_concurrent,
            "global_limit_bps": self.global_limit_bps,
            "global_available_bytes": global_available,
            "transfer_limits": {
                tid: bucket.available_tokens()
                for tid, bucket in self.transfer_buckets.items()
            }
        }


class RedisBandwidthManager:
    """
    Redis-backed bandwidth manager for distributed environments.
    Allows bandwidth management across multiple workers.
    """

    def __init__(self, redis_client: redis.Redis,
                 global_limit_bps: Optional[int] = None,
                 max_concurrent: int = 3):
        """
        Initialize Redis-backed bandwidth manager.

        Args:
            redis_client: Redis client instance
            global_limit_bps: Global bandwidth limit in bytes per second
            max_concurrent: Maximum concurrent transfers
        """
        self.redis = redis_client
        self.global_limit_bps = global_limit_bps
        self.max_concurrent = max_concurrent

        # Redis keys
        self.active_transfers_key = "bandwidth:active_transfers"
        self.global_tokens_key = "bandwidth:global_tokens"
        self.global_last_update_key = "bandwidth:global_last_update"

    def can_start_transfer(self, transfer_id: str) -> bool:
        """Check if a new transfer can start based on concurrency limit."""
        pipe = self.redis.pipeline()
        pipe.sadd(self.active_transfers_key, transfer_id)
        pipe.scard(self.active_transfers_key)
        results = pipe.execute()

        active_count = results[1]

        if active_count > self.max_concurrent:
            # Rollback
            self.redis.srem(self.active_transfers_key, transfer_id)
            return False

        return True

    def finish_transfer(self, transfer_id: str):
        """Mark transfer as finished to free concurrency slot."""
        self.redis.srem(self.active_transfers_key, transfer_id)
        # Clean up per-transfer limits
        self.redis.delete(f"bandwidth:transfer:{transfer_id}:tokens")
        self.redis.delete(f"bandwidth:transfer:{transfer_id}:last_update")

    def set_transfer_limit(self, transfer_id: str, limit_bps: int):
        """Set bandwidth limit for a specific transfer."""
        self.redis.set(f"bandwidth:transfer:{transfer_id}:limit", limit_bps)
        self.redis.set(f"bandwidth:transfer:{transfer_id}:tokens", limit_bps * 2)
        self.redis.set(f"bandwidth:transfer:{transfer_id}:last_update", time.time())

    def _refill_tokens(self, key_prefix: str, capacity: int, rate: float) -> float:
        """Refill tokens in Redis using Lua script for atomicity."""
        lua_script = """
        local tokens_key = KEYS[1]
        local last_update_key = KEYS[2]
        local capacity = tonumber(ARGV[1])
        local rate = tonumber(ARGV[2])
        local now = tonumber(ARGV[3])

        local tokens = tonumber(redis.call('GET', tokens_key) or capacity)
        local last_update = tonumber(redis.call('GET', last_update_key) or now)

        local elapsed = now - last_update
        tokens = math.min(capacity, tokens + elapsed * rate)

        redis.call('SET', tokens_key, tokens)
        redis.call('SET', last_update_key, now)

        return tokens
        """

        tokens = self.redis.eval(
            lua_script,
            2,
            f"{key_prefix}:tokens",
            f"{key_prefix}:last_update",
            capacity,
            rate,
            time.time()
        )
        return float(tokens)

    def throttle(self, transfer_id: str, bytes_to_transfer: int, timeout: Optional[float] = None) -> bool:
        """
        Throttle transfer to respect bandwidth limits using Redis.

        Args:
            transfer_id: Transfer identifier
            bytes_to_transfer: Number of bytes about to be transferred
            timeout: Maximum time to wait (seconds)

        Returns:
            True if transfer can proceed, False if timeout reached
        """
        start_time = time.time()

        while True:
            # Check global limit
            if self.global_limit_bps:
                tokens = self._refill_tokens(
                    "bandwidth:global",
                    self.global_limit_bps * 2,
                    self.global_limit_bps
                )

                if tokens >= bytes_to_transfer:
                    self.redis.decrby(f"bandwidth:global:tokens", bytes_to_transfer)
                else:
                    if timeout and (time.time() - start_time) >= timeout:
                        return False
                    time.sleep(0.01)
                    continue

            # Check per-transfer limit
            limit_bps = self.redis.get(f"bandwidth:transfer:{transfer_id}:limit")
            if limit_bps:
                limit_bps = int(limit_bps)
                tokens = self._refill_tokens(
                    f"bandwidth:transfer:{transfer_id}",
                    limit_bps * 2,
                    limit_bps
                )

                if tokens >= bytes_to_transfer:
                    self.redis.decrby(f"bandwidth:transfer:{transfer_id}:tokens", bytes_to_transfer)
                    return True
                else:
                    if timeout and (time.time() - start_time) >= timeout:
                        return False
                    time.sleep(0.01)
                    continue

            return True

    def get_stats(self) -> dict:
        """Get current bandwidth manager statistics from Redis."""
        active_count = self.redis.scard(self.active_transfers_key)

        global_available = None
        if self.global_limit_bps:
            global_available = float(self.redis.get(self.global_tokens_key) or 0)

        return {
            "active_transfers": active_count,
            "max_concurrent": self.max_concurrent,
            "global_limit_bps": self.global_limit_bps,
            "global_available_bytes": global_available,
        }


# Global bandwidth manager instance (initialized in app startup)
_bandwidth_manager: Optional[BandwidthManager] = None


def init_bandwidth_manager(global_limit_bps: Optional[int] = None, max_concurrent: int = 3):
    """Initialize global bandwidth manager."""
    global _bandwidth_manager
    _bandwidth_manager = BandwidthManager(global_limit_bps, max_concurrent)
    return _bandwidth_manager


def get_bandwidth_manager() -> BandwidthManager:
    """Get global bandwidth manager instance."""
    global _bandwidth_manager
    if _bandwidth_manager is None:
        _bandwidth_manager = BandwidthManager()
    return _bandwidth_manager
