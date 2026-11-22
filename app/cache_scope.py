"""Which cached answers a request may reuse.

A cached answer is only reused inside the same scope: tenant, chunking strategy and
retrieval mode. Different tenants never share entries, so one tenant cannot be served
an answer built for another. The demo has no login, so the tenant comes from the request
(default "public"); in a real deployment it must come from the authenticated user.

Kept free of database and numpy imports so it can be unit tested on its own.
"""
import re

DEFAULT_TENANT = "public"
TENANT_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
_TENANT_RE = re.compile(TENANT_PATTERN)


def normalise_tenant(tenant: str | None) -> str:
    if not tenant:
        return DEFAULT_TENANT
    # "|" separates the parts of a scope, so a tenant must never contain it.
    if not _TENANT_RE.fullmatch(tenant):
        raise ValueError("tenant must be 1-64 letters, digits, '-' or '_'")
    return tenant


def scope_for(strategy: str, mode: str, tenant: str | None = None) -> str:
    return f"{normalise_tenant(tenant)}|{strategy}|{mode}"
