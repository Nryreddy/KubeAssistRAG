"""
Rate Limiter Middleware.

Implements sliding-window rate limiting using Upstash Redis to protect
endpoints from abuse and enforce user quotas.
"""
import time

from upstash_redis import Redis

from app.config import settings

_redis_client: Redis | None = None

def get_redis_client() -> Redis:
    """
    Retrieve or initialize the global Upstash Redis client.
    
    Returns:
        Configured Redis client instance.
    """
    global _redis_client
    if _redis_client is None:
        _redis_client = Redis(
            url=settings.upstash_redis_url,
            token=settings.upstash_redis_token,
        )
    return _redis_client



class RateLimiter:
    """
    A sliding-window rate limiter utilizing Redis sorted sets.
    """
    def __init__(self, max_requests: int, window_seconds: int = 60):
        """
        Initialize the RateLimiter.
        
        Args:
            max_requests: The maximum number of requests allowed within the window.
            window_seconds: The duration of the sliding window in seconds.
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds

    def is_allowed(self, key: str) -> tuple[bool, int, int]:
        """
        Check if a given key has exceeded the rate limit.
        
        Uses Redis pipelining to atomically remove old requests, add the new request,
        and count total requests within the sliding window.
        
        Args:
            key: The unique identifier for the rate limit (e.g., user ID, IP + route).
            
        Returns:
            A tuple of (is_allowed, remaining_requests, total_requests_in_window).
        """
        client = get_redis_client()
        now = time.time()
        window_start = now - self.window_seconds

        pipe = client.pipeline()
        pipe.zremrangebyscore(key, 0, window_start)
        pipe.zadd(key, {str(now): now})
        pipe.zcard(key)
        pipe.expire(key, self.window_seconds)
        results = pipe.exec()

        request_count: int = results[2]  # type: ignore[assignment]
        remaining = max(0, self.max_requests - request_count)
        allowed = request_count <= self.max_requests

        return allowed, remaining, request_count

    
def is_allowed_ip(ip: str, route: str, limit: int, window_seconds: int) -> tuple[bool, int, int]:
    """
    Check rate limit for a specific IP address and route combination.
    
    Args:
        ip: The client IP address.
        route: The API route being accessed.
        limit: Max requests allowed.
        window_seconds: Sliding window duration.
        
    Returns:
        A tuple of (is_allowed, remaining_requests, request_count).
    """
    limiter = RateLimiter(max_requests=limit, window_seconds=window_seconds)
    key = f"rate_limit:ip:{ip}:{route}"
    return limiter.is_allowed(key)


def is_allowed_user(
    user_id: str, limit: int = 20, window_seconds: int = 60
) -> tuple[bool, int, int]:
    """
    Check rate limit for a specific authenticated user.
    
    Args:
        user_id: The authenticated user's ID/username.
        limit: Max requests allowed (default 20).
        window_seconds: Sliding window duration (default 60).
        
    Returns:
        A tuple of (is_allowed, remaining_requests, request_count).
    """
    limiter = RateLimiter(max_requests=limit, window_seconds=window_seconds)
    key = f"rate_limit:user:{user_id}"
    return limiter.is_allowed(key)