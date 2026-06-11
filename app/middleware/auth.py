"""
Authentication Middleware.

Provides dependency injection functions to secure FastAPI endpoints, verify JWTs,
manage user sessions, and enforce Role-Based Access Control (RBAC).
"""
import datetime

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel



from app.config import settings

security = HTTPBearer()

class User(BaseModel):
    """Pydantic model representing an authenticated user."""
    username: str
    is_admin: bool = False


def hash_password(password: str) -> str:
    """Hash a plaintext password using bcrypt."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")



def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against a bcrypt hash."""
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))




def create_access_token(
    username: str,
    expires_delta_seconds: int | None = None,
    is_admin: bool = False,
) -> str:
    """
    Generate a JSON Web Token (JWT) for a user session.
    
    Args:
        username: The authenticated user's username.
        expires_delta_seconds: Optional expiration time in seconds. Defaults to settings.
        is_admin: Flag indicating if the user has admin privileges.
        
    Returns:
        A signed JWT string.
    """
    if expires_delta_seconds is None:
        expires_delta_seconds = settings.jwt_expiration_minutes * 60
        expire = datetime.datetime.now(datetime.UTC) + datetime.timedelta(
            seconds=expires_delta_seconds
        )

    payload = {
        "sub": username,
        "exp": expire,
        "iat": datetime.datetime.now(datetime.UTC),
        "is_admin": is_admin,
    }

    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> User:
    """
    FastAPI dependency that validates a Bearer JWT and extracts the current user.
    
    Args:
        credentials: The Authorization header credentials injected by FastAPI.
        
    Returns:
        The authenticated User model.
        
    Raises:
        HTTPException: If the token is invalid or expired.
    """
    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.jwt_secret,
            algorithms=["HS256"],
        )
        username = payload.get("sub")
        if not isinstance(username, str) or username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token",
            )
        is_admin: bool = payload.get("is_admin", False)
        return User(username=username, is_admin=is_admin)
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
        ) from None
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
        ) from None


def require_admin(user: User = Depends(get_current_user)) -> User:
    """
    FastAPI dependency to enforce admin-only access on specific routes.
    
    Args:
        user: The current authenticated user object (chained dependency).
        
    Returns:
        The User object if they are an admin.
        
    Raises:
        HTTPException: 403 Forbidden if the user is not an admin.
    """
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user