"""API key check for /predict.

The key lives in the API_KEY environment variable, never in code or Git. Callers send it in the
X-API-Key header. A wrong or missing key gets 401 Unauthorized.
"""
import os
import secrets

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

# auto_error=False: we raise 401 ourselves (FastAPI's default would be 403 for a missing header).
# Using APIKeyHeader also gives /docs its "Authorize" button.
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(api_key: str = Security(api_key_header)) -> None:
    expected = os.environ.get("API_KEY")  # read on every request, so tests can set it late
    if not expected:
        # Fail closed: a server without a key configured must not serve predictions to anyone.
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Server has no API_KEY configured")

    # TODO 5: right now ANY key (or no key) is accepted. Reject the request when api_key is
    # missing OR different from `expected`:
    #   if not api_key or not secrets.compare_digest(api_key.encode(), expected.encode()):
    #       raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")
    # Check "not api_key" first: with no header, api_key is None, and compare_digest(None, ...)
    # crashes (500 instead of 401). .encode() compares bytes: compare_digest refuses text with
    # accents (another 500). Use compare_digest, not ==: it takes the same time whatever
    # the key, so nobody can guess it one character at a time from response times.
