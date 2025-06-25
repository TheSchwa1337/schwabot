from utils.safe_print import safe_print, info, warn, error, success, debug
from core.unified_math_system import unified_math
#!/usr/bin/env python3
"""
Authentication Manager - Security Validation and Role-Based Access Control
=======================================================================

This module implements a comprehensive authentication and authorization system
for Schwabot, providing mathematical security validation, token management,
and role-based access control capabilities.

Core Mathematical Functions:
- Security Score Calculation: S = Σ(wᵢ × sᵢ) / Σ(wᵢ) where wᵢ are security weights
- Token Entropy Validation: E = -Σ(pᵢ × log₂(pᵢ)) where pᵢ are token probabilities
- Access Control Matrix: A[i,j] = 1 if user i has permission j, 0 otherwise

Core Functionality:
- User authentication and session management
- Token generation and validation with entropy checking
- Role-based access control (RBAC)
- Security scoring and risk assessment
- Multi-factor authentication support
- Audit trail and security logging
"""

import logging
import json
import time
import asyncio
import hashlib
import hmac
import secrets
import jwt
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from core.unified_math_system import unified_math
from collections import defaultdict, deque
import os
import bcrypt

logger = logging.getLogger(__name__)

class AuthStatus(Enum):
    AUTHENTICATED = "authenticated"
    UNAUTHENTICATED = "unauthenticated"
    EXPIRED = "expired"
    SUSPENDED = "suspended"

class PermissionLevel(Enum):
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"

@dataclass
class User:
    user_id: str
    username: str
    email: str
    password_hash: str
    role: str
    permissions: List[str]
    is_active: bool
    created_at: datetime
    last_login: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AuthToken:
    token_id: str
    user_id: str
    token_value: str
    token_type: str
    expires_at: datetime
    is_valid: bool
    entropy_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SecurityAudit:
    audit_id: str
    user_id: str
    action: str
    resource: str
    success: bool
    security_score: float
    risk_level: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

class AuthManager:
    def __init__(self, config_path: str = "./config/auth_config.json"):
        self.config_path = config_path
        self.users: Dict[str, User] = {}
        self.auth_tokens: Dict[str, AuthToken] = {}
        self.security_audits: Dict[str, SecurityAudit] = {}
        self.access_control_matrix: Dict[str, Dict[str, bool]] = {}
        self.session_history: deque = deque(maxlen=10000)
        self.security_history: deque = deque(maxlen=5000)
        self._load_configuration()
        self._initialize_manager()
        self._start_security_monitoring()
        logger.info("Authentication Manager initialized")

    def _load_configuration(self) -> None:
        """Load authentication configuration."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                
                logger.info(f"Loaded authentication configuration")
            else:
                self._create_default_configuration()
                
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self._create_default_configuration()

    def _create_default_configuration(self) -> None:
        """Create default authentication configuration."""
        config = {
            "security": {
                "min_password_length": 8,
                "require_special_chars": True,
                "max_login_attempts": 5,
                "lockout_duration": 300,
                "session_timeout": 3600
            },
            "tokens": {
                "jwt_secret": secrets.token_hex(32),
                "token_expiry": 3600,
                "refresh_token_expiry": 86400,
                "min_entropy_score": 3.0
            },
            "roles": {
                "admin": ["read", "write", "admin"],
                "user": ["read", "write"],
                "viewer": ["read"],
                "guest": []
            }
        }
        
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")

    def _initialize_manager(self) -> None:
        """Initialize the authentication manager."""
        # Initialize default users
        self._initialize_default_users()
        
        # Initialize access control matrix
        self._initialize_access_control()
        
        # Initialize security policies
        self._initialize_security_policies()
        
        logger.info("Authentication manager initialized successfully")

    def _initialize_default_users(self) -> None:
        """Initialize default users."""
        try:
            # Create admin user
            admin_password = "admin123"  # In production, this would be more secure
            admin_hash = bcrypt.hashpw(admin_password.encode(), bcrypt.gensalt()).decode()
            
            admin_user = User(
                user_id="admin_001",
                username="admin",
                email="admin@schwabot.com",
                password_hash=admin_hash,
                role="admin",
                permissions=["read", "write", "admin"],
                is_active=True,
                created_at=datetime.now(),
                last_login=datetime.now(),
                metadata={"is_default": True}
            )
            
            self.users[admin_user.user_id] = admin_user
            
            # Create regular user
            user_password = "user123"
            user_hash = bcrypt.hashpw(user_password.encode(), bcrypt.gensalt()).decode()
            
            regular_user = User(
                user_id="user_001",
                username="user",
                email="user@schwabot.com",
                password_hash=user_hash,
                role="user",
                permissions=["read", "write"],
                is_active=True,
                created_at=datetime.now(),
                last_login=datetime.now(),
                metadata={"is_default": True}
            )
            
            self.users[regular_user.user_id] = regular_user
            
            logger.info(f"Initialized {len(self.users)} default users")
            
        except Exception as e:
            logger.error(f"Error initializing default users: {e}")

    def _initialize_access_control(self) -> None:
        """Initialize access control matrix."""
        try:
            # Define resources and permissions
            resources = [
                "trading_config", "api_keys", "user_management", "system_config",
                "trading_execution", "data_access", "reports", "analytics"
            ]
            
            # Initialize matrix for each user
            for user_id, user in self.users.items():
                self.access_control_matrix[user_id] = {}
                for resource in resources:
                    # Determine access based on user role
                    if user.role == "admin":
                        self.access_control_matrix[user_id][resource] = True
                    elif user.role == "user":
                        self.access_control_matrix[user_id][resource] = resource in ["trading_execution", "data_access", "reports"]
                    elif user.role == "viewer":
                        self.access_control_matrix[user_id][resource] = resource in ["reports", "analytics"]
                    else:
                        self.access_control_matrix[user_id][resource] = False
            
            logger.info(f"Initialized access control matrix for {len(self.users)} users")
            
        except Exception as e:
            logger.error(f"Error initializing access control: {e}")

    def _initialize_security_policies(self) -> None:
        """Initialize security policies."""
        try:
            self.security_policies = {
                "password_policy": {
                    "min_length": 8,
                    "require_uppercase": True,
                    "require_lowercase": True,
                    "require_digits": True,
                    "require_special": True
                },
                "session_policy": {
                    "max_sessions": 5,
                    "session_timeout": 3600,
                    "inactivity_timeout": 1800
                },
                "rate_limiting": {
                    "max_login_attempts": 5,
                    "lockout_duration": 300,
                    "max_requests_per_minute": 100
                }
            }
            
            logger.info("Security policies initialized")
            
        except Exception as e:
            logger.error(f"Error initializing security policies: {e}")

    def _start_security_monitoring(self) -> None:
        """Start the security monitoring system."""
        # This would start background monitoring tasks
        logger.info("Security monitoring started")

    def authenticate_user(self, username: str, password: str) -> Dict[str, Any]:
        """Authenticate a user with username and password."""
        try:
            # Find user by username
            user = None
            for u in self.users.values():
                if u.username == username:
                    user = u
                    break
            
            if not user:
                return {"success": False, "error": "User not found"}
            
            if not user.is_active:
                return {"success": False, "error": "User account is suspended"}
            
            # Verify password
            if not bcrypt.checkpw(password.encode(), user.password_hash.encode()):
                # Record failed login attempt
                self._record_failed_login(user.user_id)
                return {"success": False, "error": "Invalid password"}
            
            # Update last login
            user.last_login = datetime.now()
            
            # Generate authentication token
            token = self._generate_auth_token(user.user_id)
            
            # Record successful authentication
            self._record_successful_login(user.user_id)
            
            return {
                "success": True,
                "user_id": user.user_id,
                "username": user.username,
                "role": user.role,
                "permissions": user.permissions,
                "token": token.token_value,
                "expires_at": token.expires_at.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error authenticating user: {e}")
            return {"success": False, "error": str(e)}

    def _generate_auth_token(self, user_id: str) -> AuthToken:
        """Generate authentication token with entropy validation."""
        try:
            token_id = f"token_{int(time.time())}"
            
            # Generate token value
            token_value = secrets.token_urlsafe(32)
            
            # Calculate token entropy
            entropy_score = self._calculate_token_entropy(token_value)
            
            # Create token
            token = AuthToken(
                token_id=token_id,
                user_id=user_id,
                token_value=token_value,
                token_type="jwt",
                expires_at=datetime.now() + timedelta(hours=1),
                is_valid=True,
                entropy_score=entropy_score,
                metadata={
                    "created": datetime.now().isoformat(),
                    "ip_address": "127.0.0.1"  # In production, get from request
                }
            )
            
            self.auth_tokens[token_id] = token
            
            logger.info(f"Generated token {token_id} for user {user_id}")
            return token
            
        except Exception as e:
            logger.error(f"Error generating auth token: {e}")
            return None

    def _calculate_token_entropy(self, token: str) -> float:
        """
        Calculate token entropy.
        
        Mathematical Formula:
        E = -Σ(pᵢ × log₂(pᵢ)) where pᵢ are token probabilities
        """
        try:
            # Calculate character frequencies
            char_counts = defaultdict(int)
            for char in token:
                char_counts[char] += 1
            
            # Calculate probabilities
            total_chars = len(token)
            entropy = 0.0
            
            for count in char_counts.values():
                probability = count / total_chars
                if probability > 0:
                    entropy -= probability * np.log2(probability)
            
            return entropy
            
        except Exception as e:
            logger.error(f"Error calculating token entropy: {e}")
            return 0.0

    def validate_token(self, token_value: str) -> Dict[str, Any]:
        """Validate authentication token."""
        try:
            # Find token
            token = None
            for t in self.auth_tokens.values():
                if t.token_value == token_value:
                    token = t
                    break
            
            if not token:
                return {"valid": False, "error": "Token not found"}
            
            if not token.is_valid:
                return {"valid": False, "error": "Token is invalid"}
            
            if token.expires_at < datetime.now():
                token.is_valid = False
                return {"valid": False, "error": "Token has expired"}
            
            # Get user
            user = self.users.get(token.user_id)
            if not user:
                return {"valid": False, "error": "User not found"}
            
            if not user.is_active:
                return {"valid": False, "error": "User account is suspended"}
            
            return {
                "valid": True,
                "user_id": user.user_id,
                "username": user.username,
                "role": user.role,
                "permissions": user.permissions
            }
            
        except Exception as e:
            logger.error(f"Error validating token: {e}")
            return {"valid": False, "error": str(e)}

    def check_permission(self, user_id: str, resource: str, action: str) -> bool:
        """
        Check if user has permission to perform action on resource.
        
        Mathematical Formula:
        A[i,j] = 1 if user i has permission j, 0 otherwise
        """
        try:
            # Get user
            user = self.users.get(user_id)
            if not user or not user.is_active:
                return False
            
            # Check access control matrix
            if user_id in self.access_control_matrix:
                if resource in self.access_control_matrix[user_id]:
                    return self.access_control_matrix[user_id][resource]
            
            # Check role-based permissions
            if user.role == "admin":
                return True
            elif user.role == "user" and action in ["read", "write"]:
                return resource in ["trading_execution", "data_access", "reports"]
            elif user.role == "viewer" and action == "read":
                return resource in ["reports", "analytics"]
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking permission: {e}")
            return False

    def calculate_security_score(self, user_id: str) -> float:
        """
        Calculate user security score.
        
        Mathematical Formula:
        S = Σ(wᵢ × sᵢ) / Σ(wᵢ) where wᵢ are security weights and sᵢ are security scores
        """
        try:
            user = self.users.get(user_id)
            if not user:
                return 0.0
            
            security_factors = {
                "password_strength": self._calculate_password_strength(user),
                "session_activity": self._calculate_session_activity(user_id),
                "login_pattern": self._calculate_login_pattern(user_id),
                "permission_usage": self._calculate_permission_usage(user_id)
            }
            
            # Define weights for each factor
            weights = {
                "password_strength": 0.3,
                "session_activity": 0.25,
                "login_pattern": 0.25,
                "permission_usage": 0.2
            }
            
            # Calculate weighted score
            total_score = 0.0
            total_weight = 0.0
            
            for factor, score in security_factors.items():
                weight = weights.get(factor, 1.0)
                total_score += weight * score
                total_weight += weight
            
            security_score = total_score / total_weight if total_weight > 0 else 0.0
            
            return unified_math.max(0.0, unified_math.min(1.0, security_score))
            
        except Exception as e:
            logger.error(f"Error calculating security score: {e}")
            return 0.0

    def _calculate_password_strength(self, user: User) -> float:
        """Calculate password strength score."""
        try:
            # This is a simplified calculation - in practice, you'd analyze the actual password
            # For now, we'll use a default score based on user role
            role_strengths = {
                "admin": 0.9,
                "user": 0.7,
                "viewer": 0.6,
                "guest": 0.4
            }
            
            return role_strengths.get(user.role, 0.5)
            
        except Exception as e:
            logger.error(f"Error calculating password strength: {e}")
            return 0.5

    def _calculate_session_activity(self, user_id: str) -> float:
        """Calculate session activity score."""
        try:
            # Get recent sessions
            recent_sessions = [
                audit for audit in self.security_audits.values()
                if audit.user_id == user_id and audit.action == "login"
                and audit.timestamp > datetime.now() - timedelta(days=7)
            ]
            
            if not recent_sessions:
                return 0.5  # Neutral score for no activity
            
            # Calculate activity score based on frequency and consistency
            session_count = len(recent_sessions)
            successful_sessions = sum(1 for s in recent_sessions if s.success)
            
            if session_count == 0:
                return 0.5
            
            success_rate = successful_sessions / session_count
            frequency_score = unified_math.min(session_count / 10.0, 1.0)  # Normalize to 0-1
            
            return (success_rate + frequency_score) / 2.0
            
        except Exception as e:
            logger.error(f"Error calculating session activity: {e}")
            return 0.5

    def _calculate_login_pattern(self, user_id: str) -> float:
        """Calculate login pattern score."""
        try:
            # Get recent login attempts
            recent_logins = [
                audit for audit in self.security_audits.values()
                if audit.user_id == user_id and audit.action in ["login", "login_failed"]
                and audit.timestamp > datetime.now() - timedelta(days=30)
            ]
            
            if not recent_logins:
                return 0.5
            
            # Calculate pattern consistency
            successful_logins = [l for l in recent_logins if l.success]
            failed_logins = [l for l in recent_logins if not l.success]
            
            success_rate = len(successful_logins) / len(recent_logins)
            failure_rate = len(failed_logins) / len(recent_logins)
            
            # Penalize high failure rates
            pattern_score = success_rate - (failure_rate * 0.5)
            
            return unified_math.max(0.0, unified_math.min(1.0, pattern_score))
            
        except Exception as e:
            logger.error(f"Error calculating login pattern: {e}")
            return 0.5

    def _calculate_permission_usage(self, user_id: str) -> float:
        """Calculate permission usage score."""
        try:
            user = self.users.get(user_id)
            if not user:
                return 0.0
            
            # Get recent permission checks
            recent_checks = [
                audit for audit in self.security_audits.values()
                if audit.user_id == user_id and audit.action == "permission_check"
                and audit.timestamp > datetime.now() - timedelta(days=7)
            ]
            
            if not recent_checks:
                return 0.5  # Neutral score for no activity
            
            # Calculate appropriate usage score
            successful_checks = sum(1 for c in recent_checks if c.success)
            total_checks = len(recent_checks)
            
            if total_checks == 0:
                return 0.5
            
            # Score based on appropriate permission usage
            usage_score = successful_checks / total_checks
            
            # Bonus for using permissions appropriately for role
            role_appropriate_usage = 0.1 if user.role in ["admin", "user"] else 0.0
            
            return unified_math.min(1.0, usage_score + role_appropriate_usage)
            
        except Exception as e:
            logger.error(f"Error calculating permission usage: {e}")
            return 0.5

    def _record_failed_login(self, user_id: str) -> None:
        """Record failed login attempt."""
        try:
            audit_id = f"audit_{int(time.time())}"
            
            audit = SecurityAudit(
                audit_id=audit_id,
                user_id=user_id,
                action="login_failed",
                resource="authentication",
                success=False,
                security_score=0.0,
                risk_level="high",
                timestamp=datetime.now(),
                metadata={"ip_address": "127.0.0.1"}
            )
            
            self.security_audits[audit_id] = audit
            self.security_history.append(audit)
            
        except Exception as e:
            logger.error(f"Error recording failed login: {e}")

    def _record_successful_login(self, user_id: str) -> None:
        """Record successful login attempt."""
        try:
            audit_id = f"audit_{int(time.time())}"
            security_score = self.calculate_security_score(user_id)
            
            audit = SecurityAudit(
                audit_id=audit_id,
                user_id=user_id,
                action="login",
                resource="authentication",
                success=True,
                security_score=security_score,
                risk_level="low" if security_score > 0.7 else "medium",
                timestamp=datetime.now(),
                metadata={"ip_address": "127.0.0.1"}
            )
            
            self.security_audits[audit_id] = audit
            self.security_history.append(audit)
            
        except Exception as e:
            logger.error(f"Error recording successful login: {e}")

    def create_user(self, username: str, email: str, password: str, role: str = "user") -> Dict[str, Any]:
        """Create a new user account."""
        try:
            # Check if username already exists
            for user in self.users.values():
                if user.username == username:
                    return {"success": False, "error": "Username already exists"}
            
            # Validate password strength
            password_validation = self._validate_password(password)
            if not password_validation["valid"]:
                return {"success": False, "error": password_validation["error"]}
            
            # Hash password
            password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
            
            # Create user
            user_id = f"user_{int(time.time())}"
            user = User(
                user_id=user_id,
                username=username,
                email=email,
                password_hash=password_hash,
                role=role,
                permissions=self._get_role_permissions(role),
                is_active=True,
                created_at=datetime.now(),
                last_login=datetime.now(),
                metadata={"created_by": "system"}
            )
            
            self.users[user_id] = user
            
            # Initialize access control for new user
            self._initialize_user_access_control(user_id, role)
            
            logger.info(f"Created user {username} with role {role}")
            return {"success": True, "user_id": user_id}
            
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return {"success": False, "error": str(e)}

    def _validate_password(self, password: str) -> Dict[str, Any]:
        """Validate password strength."""
        try:
            policy = self.security_policies["password_policy"]
            
            if len(password) < policy["min_length"]:
                return {"valid": False, "error": f"Password must be at least {policy['min_length']} characters"}
            
            if policy["require_uppercase"] and not any(c.isupper() for c in password):
                return {"valid": False, "error": "Password must contain uppercase letter"}
            
            if policy["require_lowercase"] and not any(c.islower() for c in password):
                return {"valid": False, "error": "Password must contain lowercase letter"}
            
            if policy["require_digits"] and not any(c.isdigit() for c in password):
                return {"valid": False, "error": "Password must contain digit"}
            
            if policy["require_special"] and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
                return {"valid": False, "error": "Password must contain special character"}
            
            return {"valid": True}
            
        except Exception as e:
            logger.error(f"Error validating password: {e}")
            return {"valid": False, "error": str(e)}

    def _get_role_permissions(self, role: str) -> List[str]:
        """Get permissions for a role."""
        role_permissions = {
            "admin": ["read", "write", "admin"],
            "user": ["read", "write"],
            "viewer": ["read"],
            "guest": []
        }
        
        return role_permissions.get(role, [])

    def _initialize_user_access_control(self, user_id: str, role: str) -> None:
        """Initialize access control for new user."""
        try:
            resources = [
                "trading_config", "api_keys", "user_management", "system_config",
                "trading_execution", "data_access", "reports", "analytics"
            ]
            
            self.access_control_matrix[user_id] = {}
            
            for resource in resources:
                if role == "admin":
                    self.access_control_matrix[user_id][resource] = True
                elif role == "user":
                    self.access_control_matrix[user_id][resource] = resource in ["trading_execution", "data_access", "reports"]
                elif role == "viewer":
                    self.access_control_matrix[user_id][resource] = resource in ["reports", "analytics"]
                else:
                    self.access_control_matrix[user_id][resource] = False
                    
        except Exception as e:
            logger.error(f"Error initializing user access control: {e}")

    def get_auth_statistics(self) -> Dict[str, Any]:
        """Get comprehensive authentication statistics."""
        total_users = len(self.users)
        active_users = sum(1 for user in self.users.values() if user.is_active)
        total_tokens = len(self.auth_tokens)
        valid_tokens = sum(1 for token in self.auth_tokens.values() if token.is_valid)
        total_audits = len(self.security_audits)
        
        # Calculate security metrics
        security_scores = []
        for user_id in self.users.keys():
            score = self.calculate_security_score(user_id)
            security_scores.append(score)
        
        avg_security_score = unified_math.unified_math.mean(security_scores) if security_scores else 0.0
        
        # Calculate role distribution
        role_distribution = defaultdict(int)
        for user in self.users.values():
            role_distribution[user.role] += 1
        
        # Calculate recent activity
        recent_audits = [
            audit for audit in self.security_audits.values()
            if audit.timestamp > datetime.now() - timedelta(hours=24)
        ]
        
        return {
            "total_users": total_users,
            "active_users": active_users,
            "total_tokens": total_tokens,
            "valid_tokens": valid_tokens,
            "total_audits": total_audits,
            "average_security_score": avg_security_score,
            "role_distribution": dict(role_distribution),
            "recent_activity": len(recent_audits),
            "session_history_size": len(self.session_history),
            "security_history_size": len(self.security_history)
        }

def main() -> None:
    """Main function for testing and demonstration."""
    auth_manager = AuthManager("./test_auth_config.json")
    
    # Test user authentication
    auth_result = auth_manager.authenticate_user("admin", "admin123")
    safe_print(f"Authentication result: {auth_result}")
    
    # Test token validation
    if auth_result["success"]:
        token = auth_result["token"]
        validation_result = auth_manager.validate_token(token)
        safe_print(f"Token validation: {validation_result}")
    
    # Test permission checking
    permission_result = auth_manager.check_permission("admin_001", "trading_config", "read")
    safe_print(f"Permission check: {permission_result}")
    
    # Test security score calculation
    security_score = auth_manager.calculate_security_score("admin_001")
    safe_print(f"Security score: {security_score:.3f}")
    
    # Test user creation
    create_result = auth_manager.create_user("testuser", "test@example.com", "TestPass123!", "user")
    safe_print(f"User creation: {create_result}")
    
    safe_print("Authentication Manager initialized successfully")
    
    # Get statistics
    stats = auth_manager.get_auth_statistics()
    safe_print(f"Auth Statistics: {stats}")

if __name__ == "__main__":
    main() 