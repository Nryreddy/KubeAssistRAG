"""
Admin API Module.

This module provides administrative endpoints to check the health of system dependencies
and monitor/clear application caches.
"""
import asyncio
from typing import Any

from fastapi import APIRouter, Depends
from loguru import logger

from app.config import settings


from app.middleware.auth import User, require_admin
from app.services.query_cache_service import query_cache




router = APIRouter(tags=["admin"])

def _ping_postgres() -> bool:
    """
    Ping the Postgres database to check if it's reachable.

    Returns:
        True if the connection succeeds, False otherwise.
    """
    try:
        import psycopg2

        # Connect with a short timeout to prevent hanging health checks
        conn = psycopg2.connect(settings.database_url, connect_timeout=2)
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.close()
        conn.close()
        return True
    except Exception as exc:
        logger.debug("Postgres health check failed: {}", exc)
        return False

def _ping_qdrant() -> bool:
    """
    Ping the Qdrant vector store to verify its availability.

    Returns:
        True if Qdrant is reachable, False otherwise.
    """
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=settings.qdrant_url, timeout=2)
        client.get_collections()
        return True
    except Exception as exc:
        logger.debug("Qdrant health check failed: {}", exc)
        return False

def _ping_redis() -> bool:
    """
    Ping Upstash Redis to ensure caching backend is healthy.

    Returns:
        True if Redis responds to a ping, False otherwise.
    """
    try:
        from upstash_redis import Redis

        redis = Redis(url=settings.upstash_redis_url, token=settings.upstash_redis_token)
        redis.ping()
        return True
    except Exception as exc:
        logger.debug("Redis health check failed: {}", exc)
        return False

async def _ping_openai() -> bool:
    """
    Ping the OpenAI API by listing models to ensure connectivity and key validity.

    Returns:
        True if the API responds, False otherwise.
    """
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=settings.openai_api_key)
        await client.models.list()
        return True
    except Exception as exc:
        logger.debug("OpenAI health check failed: {}", exc)
        return False

def _ping_tavily() -> bool:
    """
    Verify the Tavily web search service configuration.

    Returns:
        True if the key is valid or missing (graceful degradation), False if it errors out.
    """
    try:
        from app.services.web_search import search_web

        search_web("health check")
        return True
    except ValueError:
        # Tavily key not configured — still "up" if the module loads
        return True
    except Exception as exc:
        logger.debug("Tavily health check failed: {}", exc)
        return False





@router.get("/admin/health")
async def health_check() -> dict[str, Any]:
    """
    Aggregated health check endpoint that pings all critical system dependencies.
    
    Returns:
        A dictionary containing the overall 'status' and the status of individual components.
    """
    # Concurrently ping all dependent services to gather health metrics quickly
    results = await asyncio.gather(
        asyncio.to_thread(_ping_postgres),
        asyncio.to_thread(_ping_qdrant),
        asyncio.to_thread(_ping_redis),
        _ping_openai(),
        asyncio.to_thread(_ping_tavily),
        return_exceptions=True,
    )
    postgres_ok = bool(results[0]) if not isinstance(results[0], Exception) else False
    qdrant_ok = bool(results[1]) if not isinstance(results[1], Exception) else False
    redis_ok = bool(results[2]) if not isinstance(results[2], Exception) else False
    openai_ok = bool(results[3]) if not isinstance(results[3], Exception) else False
    tavily_ok = bool(results[4]) if not isinstance(results[4], Exception) else False

    all_ok = postgres_ok and qdrant_ok and redis_ok and openai_ok and tavily_ok
    status = "ok" if all_ok else "degraded"

    return {
        "status": status,
        "qdrant": qdrant_ok,
        "postgres": postgres_ok,
        "redis": redis_ok,
        "openai": openai_ok,
        "tavily": tavily_ok
    }


@router.get("/admin/cache/stats")
async def cache_stats(user: User = Depends(require_admin)) -> dict:
    """Return per-cache hit/miss/set counts."""
    raw = query_cache.stats()

    def _tier(name: str) -> dict:
        return {
            "hits": int(raw.get(name, {}).get("hits", 0)),
            "misses": int(raw.get(name, {}).get("misses", 0)),
            "sets": int(raw.get(name, {}).get("sets", 0)),
            "hit_rate": float(raw.get(name, {}).get("hit_rate", 0.0)),
        }

    return {
        "embedding": _tier("embedding"),
        "rag": _tier("rag_answer"),
        "sql_gen": _tier("sql_gen"),
        "sql_result": _tier("sql_result"),
        "intent_router": _tier("intent"),
    }



@router.post("/admin/cache/clear")
async def cache_clear(user: User = Depends(require_admin)) -> dict:
    """Clear all caches (Redis + in-memory)."""
    cleared = query_cache.clear()
    return {"status": "ok", "cleared": cleared}