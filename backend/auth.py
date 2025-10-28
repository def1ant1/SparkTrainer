"""
JWT Authentication and Authorization Module for SparkTrainer

Implements:
- JWT access and refresh tokens
- Role-based access control (RBAC)
- Token generation and validation
- Password hashing and verification
"""

import jwt
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from functools import wraps
from flask import request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash


# Configuration
JWT_SECRET_KEY = secrets.token_urlsafe(32)  # In production, load from environment
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60  # 1 hour
REFRESH_TOKEN_EXPIRE_DAYS = 30    # 30 days


class Role:
    """User roles with permission levels"""
    ADMIN = "admin"
    MAINTAINER = "maintainer"
    VIEWER = "viewer"

    @classmethod
    def all(cls) -> List[str]:
        return [cls.ADMIN, cls.MAINTAINER, cls.VIEWER]

    @classmethod
    def validate(cls, role: str) -> bool:
        return role in cls.all()


# Role hierarchy (admin > maintainer > viewer)
ROLE_HIERARCHY = {
    Role.ADMIN: 3,
    Role.MAINTAINER: 2,
    Role.VIEWER: 1,
}


# Permission matrix: which roles can perform which actions
PERMISSIONS = {
    # Jobs
    "jobs:create": [Role.ADMIN, Role.MAINTAINER],
    "jobs:read": [Role.ADMIN, Role.MAINTAINER, Role.VIEWER],
    "jobs:cancel": [Role.ADMIN, Role.MAINTAINER],
    "jobs:delete": [Role.ADMIN],

    # Experiments
    "experiments:create": [Role.ADMIN, Role.MAINTAINER],
    "experiments:read": [Role.ADMIN, Role.MAINTAINER, Role.VIEWER],
    "experiments:update": [Role.ADMIN, Role.MAINTAINER],
    "experiments:delete": [Role.ADMIN],

    # Datasets
    "datasets:create": [Role.ADMIN, Role.MAINTAINER],
    "datasets:read": [Role.ADMIN, Role.MAINTAINER, Role.VIEWER],
    "datasets:update": [Role.ADMIN, Role.MAINTAINER],
    "datasets:delete": [Role.ADMIN],

    # Models
    "models:read": [Role.ADMIN, Role.MAINTAINER, Role.VIEWER],
    "models:tag": [Role.ADMIN, Role.MAINTAINER],
    "models:delete": [Role.ADMIN],

    # Deployments
    "deployments:create": [Role.ADMIN, Role.MAINTAINER],
    "deployments:read": [Role.ADMIN, Role.MAINTAINER, Role.VIEWER],
    "deployments:update": [Role.ADMIN, Role.MAINTAINER],
    "deployments:delete": [Role.ADMIN, Role.MAINTAINER],

    # Users & Teams
    "users:create": [Role.ADMIN],
    "users:read": [Role.ADMIN, Role.MAINTAINER],
    "users:update": [Role.ADMIN],
    "users:delete": [Role.ADMIN],

    "teams:create": [Role.ADMIN],
    "teams:read": [Role.ADMIN, Role.MAINTAINER],
    "teams:update": [Role.ADMIN],
    "teams:delete": [Role.ADMIN],

    # System
    "system:health": [Role.ADMIN, Role.MAINTAINER, Role.VIEWER],
    "system:metrics": [Role.ADMIN, Role.MAINTAINER],
}


class TokenManager:
    """Manages JWT token generation and validation"""

    @staticmethod
    def generate_access_token(user_id: str, username: str, role: str, projects: Optional[List[str]] = None) -> str:
        """Generate JWT access token"""
        if not Role.validate(role):
            raise ValueError(f"Invalid role: {role}")

        payload = {
            "user_id": user_id,
            "username": username,
            "role": role,
            "projects": projects or [],
            "type": "access",
            "exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
            "iat": datetime.utcnow(),
        }

        return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

    @staticmethod
    def generate_refresh_token(user_id: str) -> str:
        """Generate JWT refresh token"""
        payload = {
            "user_id": user_id,
            "type": "refresh",
            "exp": datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
            "iat": datetime.utcnow(),
        }

        return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

    @staticmethod
    def verify_token(token: str) -> Dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError:
            raise ValueError("Invalid token")

    @staticmethod
    def refresh_access_token(refresh_token: str, user_data: Dict[str, Any]) -> str:
        """Generate new access token from refresh token"""
        payload = TokenManager.verify_token(refresh_token)

        if payload.get("type") != "refresh":
            raise ValueError("Invalid refresh token")

        if payload.get("user_id") != user_data.get("id"):
            raise ValueError("Token user mismatch")

        return TokenManager.generate_access_token(
            user_id=user_data["id"],
            username=user_data["username"],
            role=user_data["role"],
            projects=user_data.get("projects", [])
        )


class PasswordManager:
    """Manages password hashing and verification"""

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using pbkdf2:sha256"""
        return generate_password_hash(password, method='pbkdf2:sha256')

    @staticmethod
    def verify_password(password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        return check_password_hash(password_hash, password)


class PermissionManager:
    """Manages role-based permissions"""

    @staticmethod
    def check_permission(role: str, permission: str) -> bool:
        """Check if role has permission"""
        allowed_roles = PERMISSIONS.get(permission, [])
        return role in allowed_roles

    @staticmethod
    def require_permission(permission: str):
        """Decorator to require specific permission"""
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                # Extract token from Authorization header
                auth_header = request.headers.get('Authorization')
                if not auth_header or not auth_header.startswith('Bearer '):
                    return jsonify({'error': 'Missing or invalid authorization header'}), 401

                token = auth_header.split(' ')[1]

                try:
                    payload = TokenManager.verify_token(token)
                except ValueError as e:
                    return jsonify({'error': str(e)}), 401

                # Check permission
                role = payload.get('role')
                if not PermissionManager.check_permission(role, permission):
                    return jsonify({'error': 'Insufficient permissions'}), 403

                # Add user info to request context
                request.user = payload
                return f(*args, **kwargs)

            return decorated_function
        return decorator

    @staticmethod
    def require_role(required_role: str):
        """Decorator to require specific role level or higher"""
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                # Extract token from Authorization header
                auth_header = request.headers.get('Authorization')
                if not auth_header or not auth_header.startswith('Bearer '):
                    return jsonify({'error': 'Missing or invalid authorization header'}), 401

                token = auth_header.split(' ')[1]

                try:
                    payload = TokenManager.verify_token(token)
                except ValueError as e:
                    return jsonify({'error': str(e)}), 401

                # Check role hierarchy
                user_role = payload.get('role')
                user_level = ROLE_HIERARCHY.get(user_role, 0)
                required_level = ROLE_HIERARCHY.get(required_role, 999)

                if user_level < required_level:
                    return jsonify({'error': 'Insufficient role level'}), 403

                # Add user info to request context
                request.user = payload
                return f(*args, **kwargs)

            return decorated_function
        return decorator


def require_auth(f):
    """Decorator to require authentication (any valid token)"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Extract token from Authorization header
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Missing or invalid authorization header'}), 401

        token = auth_header.split(' ')[1]

        try:
            payload = TokenManager.verify_token(token)
        except ValueError as e:
            return jsonify({'error': str(e)}), 401

        # Add user info to request context
        request.user = payload
        return f(*args, **kwargs)

    return decorated_function


# Export commonly used items
__all__ = [
    'Role',
    'TokenManager',
    'PasswordManager',
    'PermissionManager',
    'require_auth',
    'JWT_SECRET_KEY',
    'ACCESS_TOKEN_EXPIRE_MINUTES',
    'REFRESH_TOKEN_EXPIRE_DAYS',
]
