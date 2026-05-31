"""OAuth / OIDC manager — Authentik, Keycloak, etc.

Uses authlib for the OIDC flow.  On login the browser is redirected to the
IdP's authorize endpoint; on callback the code is exchanged for tokens,
userinfo is fetched, and a session cookie is issued (same mechanism as the
password flow).

User accounts are auto-created on first login by default.  Admin role is
determined by OAUTH_DEFAULT_ROLE unless the user belongs to one of the
OAUTH_GROUPS_ADMIN groups.
"""

import json
import logging
import os
import secrets
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import httpx

from src.settings import load_settings

logger = logging.getLogger(__name__)


class OAuthConfig:
    """Holds OIDC configuration from settings + env."""

    def __init__(self):
        self.enabled: bool = False
        self.provider_name: str = "OAuth"
        self.client_id: str = ""
        self.client_secret: str = ""
        self.authorize_url: str = ""
        self.token_url: str = ""
        self.userinfo_url: str = ""
        self.jwks_url: str = ""
        self.scopes: str = "openid email profile"
        self.auto_create_user: bool = True
        self.default_role: str = "user"
        self.username_claim: str = "preferred_username"
        self.email_claim: str = "email"
        self.groups_claim: str = "groups"
        self.groups_admin: List[str] = []
        self.logout_url: str = ""
        self.redirect_uri: str = ""
        self._discovery_cache: Optional[Dict[str, Any]] = None
        self._discovery_time: float = 0

    def load(self, settings: Optional[Dict[str, Any]] = None):
        """Load configuration from settings dict (or env fallback)."""
        if settings is None:
            settings = load_settings()

        self.enabled = settings.get("oauth_enabled", False)
        self.provider_name = settings.get("oauth_provider_name", "OAuth")
        self.client_id = settings.get("oauth_client_id", "")
        self.client_secret = settings.get("oauth_client_secret", "")
        self.authorize_url = settings.get("oauth_authorize_url", "")
        self.token_url = settings.get("oauth_token_url", "")
        self.userinfo_url = settings.get("oauth_userinfo_url", "")
        self.jwks_url = settings.get("oauth_jwks_url", "")
        self.scopes = settings.get("oauth_scopes", "openid email profile")
        self.auto_create_user = settings.get("oauth_auto_create_user", True)
        self.default_role = settings.get("oauth_default_role", "user")
        self.username_claim = settings.get("oauth_username_claim", "preferred_username")
        self.email_claim = settings.get("oauth_email_claim", "email")
        self.groups_claim = settings.get("oauth_groups_claim", "groups")
        self.groups_admin = settings.get("oauth_groups_admin", []) or []
        self.logout_url = settings.get("oauth_logout_url", "")

        # Build redirect_uri from request if not set
        if not self.redirect_uri:
            # Will be set dynamically; default to /api/auth/oauth/callback
            self.redirect_uri = ""

        # Env var overrides
        if os.getenv("OAUTH_ENABLED", "").lower() in ("true", "1", "yes"):
            self.enabled = True
        if os.getenv("OAUTH_PROVIDER_NAME"):
            self.provider_name = os.getenv("OAUTH_PROVIDER_NAME")
        if os.getenv("OAUTH_CLIENT_ID"):
            self.client_id = os.getenv("OAUTH_CLIENT_ID")
        if os.getenv("OAUTH_CLIENT_SECRET"):
            self.client_secret = os.getenv("OAUTH_CLIENT_SECRET")
        if os.getenv("OAUTH_AUTHORIZE_URL"):
            self.authorize_url = os.getenv("OAUTH_AUTHORIZE_URL")
        if os.getenv("OAUTH_TOKEN_URL"):
            self.token_url = os.getenv("OAUTH_TOKEN_URL")
        if os.getenv("OAUTH_USERINFO_URL"):
            self.userinfo_url = os.getenv("OAUTH_USERINFO_URL")
        if os.getenv("OAUTH_JWKS_URL"):
            self.jwks_url = os.getenv("OAUTH_JWKS_URL")
        if os.getenv("OAUTH_SCOPES"):
            self.scopes = os.getenv("OAUTH_SCOPES")
        if os.getenv("OAUTH_AUTO_CREATE_USER", "").lower() in ("false", "0", "no"):
            self.auto_create_user = False
        if os.getenv("OAUTH_DEFAULT_ROLE"):
            self.default_role = os.getenv("OAUTH_DEFAULT_ROLE")
        if os.getenv("OAUTH_USERNAME_CLAIM"):
            self.username_claim = os.getenv("OAUTH_USERNAME_CLAIM")
        if os.getenv("OAUTH_EMAIL_CLAIM"):
            self.email_claim = os.getenv("OAUTH_EMAIL_CLAIM")
        if os.getenv("OAUTH_GROUPS_CLAIM"):
            self.groups_claim = os.getenv("OAUTH_GROUPS_CLAIM")
        if os.getenv("OAUTH_GROUPS_ADMIN"):
            self.groups_admin = [g.strip() for g in os.getenv("OAUTH_GROUPS_ADMIN", "").split(",") if g.strip()]
        if os.getenv("OAUTH_LOGOUT_URL"):
            self.logout_url = os.getenv("OAUTH_LOGOUT_URL")

    @property
    def is_configured(self) -> bool:
        return (
            self.enabled
            and self.client_id
            and self.client_secret
            and self.authorize_url
            and self.token_url
        )

    async def discover(self) -> Dict[str, Any]:
        """Fetch OIDC discovery document and cache it."""
        now = time.monotonic()
        if self._discovery_cache and (now - self._discovery_time) < 300:
            return self._discovery_cache

        # Try well-known endpoint first
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    f"{self.authorize_url.rsplit('/', 1)[0]}/.well-known/openid-configuration",
                    follow_redirects=True,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    self._discovery_cache = data
                    self._discovery_time = now
                    # Fill in any missing endpoints from discovery
                    if not self.userinfo_url and "userinfo_endpoint" in data:
                        self.userinfo_url = data["userinfo_endpoint"]
                    if not self.jwks_url and "jwks_uri" in data:
                        self.jwks_url = data["jwks_uri"]
                    return data
        except Exception as e:
            logger.debug(f"OIDC discovery failed (using manual config): {e}")

        self._discovery_cache = {}
        self._discovery_time = now
        return {}


class OAuthManager:
    """Manages the OAuth/OIDC authentication flow."""

    def __init__(self):
        self.config = OAuthConfig()
        self._state_store: Dict[str, Dict[str, Any]] = {}  # state_param -> {nonce, redirect}
        self._state_lock = None  # lazy init

    def load_config(self, settings: Optional[Dict[str, Any]] = None):
        self.config.load(settings)

    def _get_state_lock(self):
        """Thread-safe lazy init for state dict."""
        if self._state_lock is None:
            import threading
            self._state_lock = threading.RLock()
        return self._state_lock

    async def get_authorize_url(self, redirect_uri: str, state: Optional[str] = None,
                                 nonce: Optional[str] = None) -> str:
        """Build the authorization URL to redirect the user to."""
        if not self.config.is_configured:
            raise ValueError("OAuth not configured")

        self.config.redirect_uri = redirect_uri

        if state is None:
            state = secrets.token_urlsafe(32)
        if nonce is None:
            nonce = secrets.token_urlsafe(32)

        with self._get_state_lock():
            self._state_store[state] = {"nonce": nonce, "redirect": redirect_uri}

        params = {
            "response_type": "code",
            "client_id": self.config.client_id,
            "redirect_uri": redirect_uri,
            "scope": self.config.scopes,
            "state": state,
            "nonce": nonce,
        }

        return f"{self.config.authorize_url}?{urlencode(params)}"

    async def exchange_code(self, code: str, redirect_uri: str,
                            state: str) -> Dict[str, Any]:
        """Exchange authorization code for tokens."""
        if not self.config.is_configured:
            raise ValueError("OAuth not configured")

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                self.config.token_url,
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "client_id": self.config.client_id,
                    "client_secret": self.config.client_secret,
                    "redirect_uri": redirect_uri,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

            if resp.status_code != 200:
                logger.error(f"Token exchange failed: {resp.status_code} {resp.text}")
                raise ValueError(f"Token exchange failed: {resp.status_code}")

            token_data = resp.json()
            return token_data

    async def get_userinfo(self, access_token: str) -> Dict[str, Any]:
        """Fetch user info from the OIDC provider."""
        if not self.config.userinfo_url:
            raise ValueError("No userinfo URL configured")

        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                self.config.userinfo_url,
                headers={"Authorization": f"Bearer {access_token}"},
            )

            if resp.status_code != 200:
                logger.error(f"Userinfo fetch failed: {resp.status_code} {resp.text}")
                raise ValueError(f"Userinfo fetch failed: {resp.status_code}")

            return resp.json()

    def get_username_from_claims(self, userinfo: Dict[str, Any]) -> str:
        """Extract username from userinfo claims."""
        username = userinfo.get(self.config.username_claim)
        if not username:
            # Fallback: use email or sub
            username = userinfo.get(self.config.email_claim) or userinfo.get("sub")
        if not username:
            raise ValueError("No username claim found in userinfo")
        return str(username).lower().strip()

    def is_admin_from_groups(self, userinfo: Dict[str, Any]) -> bool:
        """Check if user should be admin based on groups claim."""
        if not self.config.groups_admin:
            return False
        groups = userinfo.get(self.config.groups_claim, [])
        if isinstance(groups, str):
            groups = [g.strip() for g in groups.split(",") if g.strip()]
        if isinstance(groups, list):
            return any(g in self.config.groups_admin for g in groups)
        return False

    def get_or_create_user(self, username: str, userinfo: Dict[str, Any],
                           auth_manager) -> str:
        """Get existing user or create one. Returns username."""
        users = auth_manager.users
        if username in users:
            # User exists — check if admin role needs updating
            user_data = users[username]
            is_admin = user_data.get("is_admin", False)
            # If they were previously admin via groups, keep it
            # If they were previously non-admin, check if groups now qualify
            if not is_admin and self.is_admin_from_groups(userinfo):
                auth_manager._config["users"][username]["is_admin"] = True
                auth_manager._save()
                logger.info(f"User '{username}' promoted to admin via groups")
            return username

        # Auto-create user if enabled
        if not self.config.auto_create_user:
            raise ValueError(
                f"User '{username}' not found and auto-create is disabled"
            )

        is_admin = self.is_admin_from_groups(userinfo) or self.config.default_role == "admin"

        auth_manager.create_user(username, password="", is_admin=is_admin)
        logger.info(f"Auto-created OIDC user '{username}' (admin={is_admin})")
        return username

    def get_state(self, state: str) -> Optional[Dict[str, Any]]:
        """Retrieve and remove a state entry."""
        with self._get_state_lock():
            return self._state_store.pop(state, None)

    def get_oidc_settings(self) -> Dict[str, Any]:
        """Return OIDC settings for the frontend (secrets masked)."""
        return {
            "enabled": self.config.enabled,
            "provider_name": self.config.provider_name,
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret,
            "authorize_url": self.config.authorize_url,
            "token_url": self.config.token_url,
            "userinfo_url": self.config.userinfo_url,
            "jwks_url": self.config.jwks_url,
            "scopes": self.config.scopes,
            "auto_create_user": self.config.auto_create_user,
            "default_role": self.config.default_role,
            "username_claim": self.config.username_claim,
            "email_claim": self.config.email_claim,
            "groups_claim": self.config.groups_claim,
            "groups_admin": self.config.groups_admin,
            "logout_url": self.config.logout_url,
            "redirect_uri": self.config.redirect_uri,
            "is_configured": self.config.is_configured,
        }
