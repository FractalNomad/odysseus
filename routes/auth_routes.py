"""Authentication routes — login, logout, signup, status, user management."""

from fastapi import APIRouter, Request, Response, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import Optional
import logging
import os

from core.auth import AuthManager
from core.oauth import OAuthManager
from src.rate_limiter import RateLimiter
from src.settings import (
    load_settings as _load_settings,
    save_settings as _save_settings,
    load_features as _load_features,
    save_features as _save_features,
    DEFAULT_SETTINGS,
)
from src.integrations import (
    load_integrations,
    add_integration,
    update_integration,
    delete_integration,
    get_integration,
    execute_api_call,
    INTEGRATION_PRESETS,
    migrate_from_settings,
)

logger = logging.getLogger(__name__)

# Module-level reference for OAuthManager (set by setup_auth_routes)
_oauth_manager: Optional[OAuthManager] = None


class LoginRequest(BaseModel):
    username: str
    password: str
    remember: bool = True
    totp_code: Optional[str] = None


class SetupRequest(BaseModel):
    username: str
    password: str


class SignupRequest(BaseModel):
    username: str
    password: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


class CreateUserRequest(BaseModel):
    username: str
    password: str
    is_admin: bool = False


class DeleteUserRequest(BaseModel):
    username: str


SESSION_COOKIE = "odysseus_session"


def setup_auth_routes(auth_manager: AuthManager, oauth_manager: Optional[OAuthManager] = None) -> APIRouter:
    router = APIRouter(prefix="/api/auth", tags=["auth"])

    # Load OIDC config
    if oauth_manager is None:
        oauth_manager = OAuthManager()
        oauth_manager.load_config()
    global _oauth_manager
    _oauth_manager = oauth_manager

    _login_limiter = RateLimiter(max_requests=15, window_seconds=60)
    _signup_limiter = RateLimiter(max_requests=3, window_seconds=300)
    _setup_limiter = RateLimiter(max_requests=3, window_seconds=300)

    def _get_current_user(request: Request) -> Optional[str]:
        token = request.cookies.get(SESSION_COOKIE)
        return auth_manager.get_username_for_token(token)

    @router.post("/setup")
    async def first_run_setup(body: SetupRequest, request: Request):
        """Create initial admin account. Only works if no accounts exist."""
        if not _setup_limiter.check(request.client.host):
            raise HTTPException(429, "Too many requests — try again later")
        if auth_manager.is_configured:
            raise HTTPException(400, "Already configured")
        if len(body.password) < 8:
            raise HTTPException(400, "Password must be at least 8 characters")
        ok = auth_manager.setup(body.username, body.password)
        if not ok:
            raise HTTPException(500, "Setup failed")
        return {"ok": True, "message": "Admin account created"}

    @router.post("/signup")
    async def signup(body: SignupRequest, request: Request):
        """Create a new user account. Only works if signup is enabled by admin."""
        if not _signup_limiter.check(request.client.host):
            raise HTTPException(429, "Too many requests — try again later")
        if not auth_manager.is_configured:
            raise HTTPException(400, "Run setup first")
        if not auth_manager.signup_enabled:
            raise HTTPException(403, "Registration is disabled. Ask an admin for an account.")
        if len(body.password) < 8:
            raise HTTPException(400, "Password must be at least 8 characters")
        if len(body.username.strip()) < 1:
            raise HTTPException(400, "Username is required")
        ok = auth_manager.create_user(body.username, body.password, is_admin=False)
        if not ok:
            raise HTTPException(409, "Username already taken")
        return {"ok": True, "message": "Account created"}

    @router.post("/login")
    async def login(body: LoginRequest, request: Request, response: Response):
        if not _login_limiter.check(request.client.host):
            raise HTTPException(429, "Too many requests — try again later")
        # Verify password first
        username = body.username.strip().lower()
        if not auth_manager.verify_password(username, body.password):
            raise HTTPException(401, "Invalid credentials")
        # Check 2FA if enabled
        if auth_manager.totp_enabled(username):
            if not body.totp_code:
                # Password OK but need TOTP — tell client to show code input
                return {"ok": False, "requires_totp": True, "username": username}
            if not auth_manager.totp_verify(username, body.totp_code):
                raise HTTPException(401, "Invalid 2FA code")
        # All checks passed — create session
        token = auth_manager.create_session(username, body.password)
        if not token:
            raise HTTPException(401, "Invalid credentials")
        cookie_kwargs = dict(
            key=SESSION_COOKIE,
            value=token,
            httponly=True,
            samesite="lax",
            secure=os.getenv("SECURE_COOKIES", "false").lower() == "true",
            path="/",
        )
        if body.remember:
            cookie_kwargs["max_age"] = 60 * 60 * 24 * 7  # 7 days
        response.set_cookie(**cookie_kwargs)
        return {"ok": True, "username": username}

    @router.post("/logout")
    async def logout(request: Request, response: Response):
        token = request.cookies.get(SESSION_COOKIE)
        if token:
            auth_manager.revoke_token(token)
        response.delete_cookie(SESSION_COOKIE, path="/")
        return {"ok": True}

    @router.get("/status")
    async def auth_status(request: Request):
        token = request.cookies.get(SESSION_COOKIE)
        result = auth_manager.status(token)
        result["signup_enabled"] = auth_manager.signup_enabled
        # Include the caller's effective privileges so the frontend can
        # hide / dim UI controls the user isn't allowed to use. Admins get
        # ADMIN_PRIVILEGES (everything on), regular users get their stored
        # set merged with DEFAULT_PRIVILEGES.
        try:
            u = result.get("username")
            if u:
                result["privileges"] = auth_manager.get_privileges(u)
        except Exception:
            pass
        return result

    @router.post("/change-password")
    async def change_password(body: ChangePasswordRequest, request: Request):
        user = _get_current_user(request)
        if not user:
            raise HTTPException(401, "Not authenticated")
        if len(body.new_password) < 8:
            raise HTTPException(400, "Password must be at least 8 characters")
        ok = auth_manager.change_password(user, body.current_password, body.new_password)
        if not ok:
            raise HTTPException(400, "Current password is incorrect")
        return {"ok": True}

    # ------------------------------------------------------------------
    # Two-factor authentication
    # ------------------------------------------------------------------

    @router.post("/2fa/setup")
    async def totp_setup(request: Request):
        """Generate a TOTP secret and return the QR code URI."""
        user = _get_current_user(request)
        if not user:
            raise HTTPException(401, "Not authenticated")
        if auth_manager.totp_enabled(user):
            raise HTTPException(400, "2FA is already enabled")
        secret = auth_manager.totp_generate_secret(user)
        if not secret:
            raise HTTPException(500, "Failed to generate secret")
        uri = auth_manager.totp_get_provisioning_uri(user, secret)
        # Generate QR code as base64 PNG
        import qrcode, io, base64
        qr = qrcode.make(uri, box_size=6, border=2)
        buf = io.BytesIO()
        qr.save(buf, format="PNG")
        qr_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return {"secret": secret, "uri": uri, "qr_code": f"data:image/png;base64,{qr_b64}"}

    class TotpVerifyRequest(BaseModel):
        code: str

    @router.post("/2fa/confirm")
    async def totp_confirm(body: TotpVerifyRequest, request: Request):
        """Verify a TOTP code to confirm 2FA setup. Returns backup codes."""
        user = _get_current_user(request)
        if not user:
            raise HTTPException(401, "Not authenticated")
        if not auth_manager.totp_confirm_enable(user, body.code):
            raise HTTPException(400, "Invalid code — try again")
        backup = auth_manager.users.get(user, {}).get("totp_backup_codes", [])
        return {"ok": True, "backup_codes": backup}

    class TotpDisableRequest(BaseModel):
        password: str

    @router.post("/2fa/disable")
    async def totp_disable(body: TotpDisableRequest, request: Request):
        """Disable 2FA. Requires password confirmation."""
        user = _get_current_user(request)
        if not user:
            raise HTTPException(401, "Not authenticated")
        if not auth_manager.totp_disable(user, body.password):
            raise HTTPException(400, "Invalid password")
        return {"ok": True}

    @router.get("/2fa/status")
    async def totp_status(request: Request):
        """Check if 2FA is enabled for the current user."""
        user = _get_current_user(request)
        if not user:
            raise HTTPException(401, "Not authenticated")
        return {"enabled": auth_manager.totp_enabled(user)}

    # Admin-only routes
    @router.get("/users")
    async def list_users(request: Request):
        user = _get_current_user(request)
        if not user or not auth_manager.is_admin(user):
            raise HTTPException(403, "Admin only")
        return {"users": auth_manager.list_users()}

    @router.post("/users")
    async def admin_create_user(body: CreateUserRequest, request: Request):
        user = _get_current_user(request)
        if not user or not auth_manager.is_admin(user):
            raise HTTPException(403, "Admin only")
        if len(body.password) < 8:
            raise HTTPException(400, "Password must be at least 8 characters")
        ok = auth_manager.create_user(body.username, body.password, body.is_admin)
        if not ok:
            raise HTTPException(409, "Username already taken")
        return {"ok": True}

    @router.put("/users/{username}/privileges")
    async def update_user_privileges(username: str, request: Request):
        user = _get_current_user(request)
        if not user or not auth_manager.is_admin(user):
            raise HTTPException(403, "Admin only")
        body = await request.json()
        ok = auth_manager.set_privileges(username, body)
        if not ok:
            raise HTTPException(404, "User not found or is admin")
        return {"ok": True, "privileges": auth_manager.get_privileges(username)}

    @router.post("/signup-toggle")
    async def toggle_signup(request: Request):
        """Toggle open registration on/off. Admin only."""
        user = _get_current_user(request)
        if not user or not auth_manager.is_admin(user):
            raise HTTPException(403, "Admin only")
        auth_manager.signup_enabled = not auth_manager.signup_enabled
        return {"ok": True, "signup_enabled": auth_manager.signup_enabled}

    @router.delete("/users")
    async def admin_delete_user(body: DeleteUserRequest, request: Request):
        user = _get_current_user(request)
        if not user or not auth_manager.is_admin(user):
            raise HTTPException(403, "Admin only")
        ok = auth_manager.delete_user(body.username, user)
        if not ok:
            raise HTTPException(400, "Cannot delete user")
        return {"ok": True}

    # ---- Feature visibility (admin-managed) ----

    @router.get("/features")
    async def get_features():
        """Public: returns which UI features are enabled."""
        return _load_features()

    @router.post("/features")
    async def set_features(request: Request):
        """Admin only: update feature toggles."""
        user = _get_current_user(request)
        if not user or not auth_manager.is_admin(user):
            raise HTTPException(403, "Admin only")
        body = await request.json()
        current = _load_features()
        for key in current:
            if key in body and isinstance(body[key], bool):
                current[key] = body[key]
        _save_features(current)
        return current

    # ---- App settings (admin-managed) ----

    _SECRET_KEY_PATTERNS = ("_api_key", "_password", "_secret", "_token", "_key")

    def _is_secret_key(name: str) -> bool:
        n = (name or "").lower()
        if n in ("google_pse_cx",):  # public identifier, not a secret
            return False
        return any(n.endswith(p) or n == p.lstrip("_") for p in _SECRET_KEY_PATTERNS)

    def _scrub_settings(settings: dict) -> dict:
        """Return a copy of settings with secret-shaped values masked.

        Frontend reads /settings without auth for things like keybinds + TTS
        prefs. Secrets (search-provider keys, IMAP/SMTP passwords) must NOT
        be exposed to non-admin callers.
        """
        scrubbed = {}
        for k, v in (settings or {}).items():
            if _is_secret_key(k) and isinstance(v, str) and v:
                scrubbed[k] = ""  # presence preserved, value blanked
            else:
                scrubbed[k] = v
        return scrubbed

    @router.get("/settings")
    async def get_settings(request: Request):
        """Returns app settings. Admins get the full set; non-admins get
        a scrubbed copy with secret keys blanked. The frontend uses this
        for keybinds + TTS prefs, so it stays callable without admin."""
        user = _get_current_user(request)
        settings = _load_settings()
        if user and auth_manager.is_admin(user):
            return settings
        return _scrub_settings(settings)

    @router.post("/settings")
    async def set_settings(request: Request):
        """Admin only: update app settings."""
        user = _get_current_user(request)
        if not user or not auth_manager.is_admin(user):
            raise HTTPException(403, "Admin only")
        body = await request.json()
        current = _load_settings()
        for key in DEFAULT_SETTINGS:
            if key in body:
                current[key] = body[key]
        _save_settings(current)
        return current

    # ---- Integrations CRUD ----

    # Run migration on startup
    migrate_from_settings()

    @router.get("/integrations")
    async def list_integrations_route(request: Request):
        """List all integrations (admin only, keys masked)."""
        user = _get_current_user(request)
        if not user or not auth_manager.is_admin(user):
            raise HTTPException(403, "Admin only")
        items = load_integrations()
        # Mask API keys for frontend display
        safe = []
        for item in items:
            copy = dict(item)
            if copy.get("api_key"):
                copy["api_key"] = copy["api_key"][:4] + "****"
            safe.append(copy)
        return {"integrations": safe}

    @router.get("/integrations/presets")
    async def list_presets():
        """List available integration presets."""
        return {"presets": {k: {kk: vv for kk, vv in v.items() if kk != "api_key"} for k, v in INTEGRATION_PRESETS.items()}}

    @router.post("/integrations")
    async def create_integration(request: Request):
        """Create a new integration (admin only)."""
        user = _get_current_user(request)
        if not user or not auth_manager.is_admin(user):
            raise HTTPException(403, "Admin only")
        body = await request.json()
        item = add_integration(body)
        return {"ok": True, "integration": item}

    @router.put("/integrations/{integration_id}")
    async def update_integration_route(integration_id: str, request: Request):
        """Update an existing integration (admin only)."""
        user = _get_current_user(request)
        if not user or not auth_manager.is_admin(user):
            raise HTTPException(403, "Admin only")
        body = await request.json()
        item = update_integration(integration_id, body)
        if not item:
            raise HTTPException(404, "Integration not found")
        return {"ok": True, "integration": item}

    @router.delete("/integrations/{integration_id}")
    async def delete_integration_route(integration_id: str, request: Request):
        """Delete an integration (admin only)."""
        user = _get_current_user(request)
        if not user or not auth_manager.is_admin(user):
            raise HTTPException(403, "Admin only")
        ok = delete_integration(integration_id)
        if not ok:
            raise HTTPException(404, "Integration not found")
        return {"ok": True}

    @router.post("/integrations/{integration_id}/test")
    async def test_integration_route(integration_id: str, request: Request):
        """Test connectivity to an integration (admin only)."""
        user = _get_current_user(request)
        if not user or not auth_manager.is_admin(user):
            raise HTTPException(403, "Admin only")
        integ = get_integration(integration_id)
        if not integ:
            raise HTTPException(404, "Integration not found")
        preset = (integ.get("preset") or integ.get("name", "")).lower()

        # ntfy is special: a GET / proves the server is reachable but
        # publishes nothing, so the user has no way to know whether
        # subscribers will actually receive notifications. Instead, do
        # the real thing — POST a one-line "connectivity test" message
        # to the topic the Reminders panel is configured to use. If the
        # subscriber app is wired up correctly, this is what the green
        # checkmark + a phone ping confirms together.
        if preset == "ntfy":
            import httpx
            from urllib.parse import urlparse
            # Strip any path/query the user accidentally pasted in the
            # base URL (e.g. `http://host:8091/odysseus`) — otherwise
            # the topic gets appended after the path and we publish to
            # `/odysseus/odysseus` (which ntfy 404s on). ntfy itself
            # only ever serves from the root.
            raw_base = (integ.get("base_url") or "").strip()
            parsed = urlparse(raw_base)
            base = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else raw_base.rstrip("/")
            settings = _load_settings()
            topic = (settings.get("reminder_ntfy_topic") or "reminders").strip() or "reminders"
            full_url = f"{base}/{topic}"
            api_key = integ.get("api_key", "")
            auth_type = (integ.get("auth_type") or "none").lower()
            headers = {
                "Title": "Odysseus connectivity test",
                "Tags": "white_check_mark",
                "Priority": "default",
            }
            if api_key:
                if auth_type == "bearer":
                    headers["Authorization"] = f"Bearer {api_key}"
                elif auth_type == "header":
                    headers[integ.get("auth_header") or "Authorization"] = api_key
            try:
                async with httpx.AsyncClient(timeout=8.0) as client:
                    r = await client.post(
                        full_url,
                        content="Connectivity test from Odysseus. If you see this on your phone, ntfy is wired up correctly.",
                        headers=headers,
                    )
                if r.is_success:
                    # Tell the user EXACTLY where it went and what to
                    # subscribe to on their phone, so they can match
                    # without guesswork. The doubled-topic / wrong-host
                    # mistakes are easier to spot when the actual URL
                    # is right there in the success line.
                    return {
                        "ok": True,
                        "message": (
                            f"Sent to {full_url} — on your ntfy app, "
                            f"subscribe to topic \"{topic}\" with server "
                            f"\"{base}\" (or paste the full URL: {full_url})."
                        ),
                    }
                return {"ok": False, "message": f"ntfy returned HTTP {r.status_code} from {full_url}: {r.text[:200]}"}
            except Exception as e:
                return {"ok": False, "message": f"ntfy publish to {full_url} failed: {e}"[:300]}

        # All other presets: GET against a known health endpoint.
        # Fall back to detecting from name if preset is missing.
        health_paths = {
            "miniflux": "/v1/me",
            "gitea": "/api/v1/version",
            "linkding": "/api/tags/",
            "homeassistant": "/api/",
            "home assistant": "/api/",
        }
        path = health_paths.get(preset, "/")
        result = await execute_api_call(integration_id, "GET", path)
        if result.get("exit_code", 1) == 0:
            return {"ok": True, "message": "Connection successful"}
        return {"ok": False, "message": (result.get("error") or "Connection failed")[:300]}

    # ---- OAuth / OIDC routes ----

    @router.get("/oidc/settings")
    async def get_oidc_settings():
        """Return OIDC settings for the frontend (public)."""
        om: OAuthManager = _oauth_manager
        if om is None:
            return {"enabled": False, "is_configured": False}
        return om.get_oidc_settings()

    @router.get("/oauth/login")
    async def oauth_login(request: Request, redirect: str = ""):
        """Redirect to the IdP's authorize endpoint."""
        om: OAuthManager = _oauth_manager
        if om is None or not om.config.is_configured:
            raise HTTPException(400, "OAuth not configured")

        # Auto-discover endpoints if discovery_url is set
        if om.config.discovery_url:
            await om.config.discover()

        # Build the redirect_uri from the request
        redirect_uri = str(request.url_for("oauth_callback"))

        try:
            authorize_url = await om.get_authorize_url(redirect_uri)
        except ValueError as e:
            raise HTTPException(400, str(e))

        return RedirectResponse(url=authorize_url, status_code=302)

    @router.get("/oauth/callback")
    async def oauth_callback(request: Request, response: Response,
                             code: str = "", state: str = "", error: str = ""):
        """Handle the OAuth callback from the IdP."""
        om: OAuthManager = _oauth_manager
        if om is None or not om.config.is_configured:
            return RedirectResponse(url="/login", status_code=302)

        if error:
            # IdP returned an error
            err_desc = request.query_params.get("error_description", error)
            return RedirectResponse(
                url=f"/login?oidc_error={err_desc}", status_code=302
            )

        if not code or not state:
            return RedirectResponse(url="/login", status_code=302)

        # Validate state
        state_data = om.get_state(state)
        if not state_data:
            return RedirectResponse(
                url="/login?oidc_error=Invalid or expired state parameter",
                status_code=302,
            )

        # Use base URL without query params — Authentik validates that this
        # redirect_uri matches the one from the authorization request.
        redirect_uri = str(request.url).split("?")[0]

        try:
            logger.info(f"OAuth callback: code={code[:10]}..., state={state[:10]}..., redirect_uri={redirect_uri}")
            # Exchange code for tokens
            token_data = await om.exchange_code(code, redirect_uri, state)
            access_token = token_data.get("access_token", "")
            id_token = token_data.get("id_token", "")

            if not access_token:
                return RedirectResponse(
                    url="/login?oidc_error=No access token received",
                    status_code=302,
                )

            # Fetch userinfo
            userinfo = await om.get_userinfo(access_token)
            logger.info(f"OAuth callback: userinfo={userinfo}")

            # Extract username
            username = om.get_username_from_claims(userinfo)
            logger.info(f"OAuth callback: username={username}")

            # Get or create user (uses OAuth config for first_user_admin, etc.)
            om.get_or_create_user(
                username, userinfo, auth_manager
            )

            # Create session (use a dummy password — session is OIDC-based)
            logger.info(f"OAuth callback: creating session for {username}")
            token = auth_manager.create_session(username, "oidc")
            logger.info(f"OAuth callback: session token={token}")
            if not token:
                return RedirectResponse(
                    url="/login?oidc_error=Session creation failed",
                    status_code=302,
                )

            # Store OIDC metadata in session for later use
            session_path = os.path.join(
                os.path.dirname(auth_manager.auth_path), "sessions.json"
            )
            try:
                import json as _json
                with open(session_path, "r") as f:
                    sessions = _json.load(f)
                if token in sessions:
                    sessions[token]["oidc"] = True
                    sessions[token]["oidc_username"] = username
                with open(session_path, "w") as f:
                    _json.dump(sessions, f)
            except Exception:
                pass

            # Set session cookie on the redirect response
            redirect = RedirectResponse(url="/", status_code=302)
            redirect.set_cookie(
                key=SESSION_COOKIE,
                value=token,
                httponly=True,
                samesite="lax",
                secure=os.getenv("SECURE_COOKIES", "false").lower() == "true",
                path="/",
                max_age=60 * 60 * 24 * 7,  # 7 days
            )
            return redirect

        except ValueError as e:
            return RedirectResponse(
                url=f"/login?oidc_error={str(e)}",
                status_code=302,
            )
        except Exception as e:
            logger.error(f"OAuth callback error: {e}", exc_info=True)
            return RedirectResponse(
                url="/login?oidc_error=Authentication failed",
                status_code=302,
            )

    @router.get("/oauth/logout")
    async def oauth_logout(request: Request, response: Response):
        """Log out and optionally redirect to IdP logout."""
        om: OAuthManager = _oauth_manager
        token = request.cookies.get(SESSION_COOKIE)
        if token:
            auth_manager.revoke_token(token)
        response.delete_cookie(SESSION_COOKIE, path="/")

        # Redirect to IdP logout if configured
        if om and om.config.logout_url:
            return RedirectResponse(url=om.config.logout_url, status_code=302)
        return RedirectResponse(url="/login", status_code=302)

    return router
