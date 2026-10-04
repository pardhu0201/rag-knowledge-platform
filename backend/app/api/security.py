"""Opt-in protection for endpoints that change the shared knowledge base.

The public demo is deliberately open so anyone can try uploading a document.
Set ``ADMIN_TOKEN`` and document upload and delete then require an
``X-Admin-Token`` header with that value - otherwise any visitor could delete
the corpus or plant a document carrying an indirect prompt injection.
Querying stays open.

This is a single shared secret, not user authentication: a real deployment
would put SSO in front of the API and scope documents per user or team.
"""

from __future__ import annotations

import hmac

from fastapi import Header, HTTPException

from app.config import settings


def require_admin(x_admin_token: str | None = Header(default=None)) -> None:
    expected = settings.admin_token.strip()
    if not expected:
        return  # open demo mode
    # Constant-time comparison so the token cannot be recovered by timing.
    if not x_admin_token or not hmac.compare_digest(x_admin_token.strip(), expected):
        raise HTTPException(status_code=401, detail="Admin token required")
