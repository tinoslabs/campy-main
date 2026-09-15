"""Putting camera credentials into a stream URL without breaking it.

A stream URL and a username and password look like three independent fields,
but they end up in one string, and the rules for that string are stricter than
they look. Two ways to get it wrong both end in the same 401 Unauthorized from
the camera, with nothing to say which:

* The URL already carries credentials, because that is the form the vendor's
  app and manual give you — and the fields get filled in as well. Injecting
  again sends "admin:secret@admin:secret" as the password.
* A password contains a character that means something in a URL. "pa/ss" ends
  the authority early and the host becomes the username.
"""
from __future__ import annotations

from urllib.parse import quote, urlsplit

#: Everything RFC 3986 permits unescaped in userinfo. Anything else is encoded.
_USERINFO_SAFE = "-._~!$&'()*+,;="


def has_credentials(url: str) -> bool:
    """Does this URL already carry a username (and possibly a password)?"""
    if not url or "://" not in url:
        return False
    try:
        return bool(urlsplit(url).username)
    except ValueError:
        return "@" in url.split("://", 1)[1].split("/", 1)[0]


def split_credentials(url: str) -> tuple[str, str, str]:
    """Pull credentials out of a URL, returning (clean url, username, password)."""
    if not has_credentials(url):
        return url, "", ""
    scheme, rest = url.split("://", 1)
    authority, _, tail = rest.partition("/")
    userinfo, _, host = authority.rpartition("@")
    username, _, password = userinfo.partition(":")
    clean = f"{scheme}://{host}" + (f"/{tail}" if tail else "")
    return clean, username, password


def inject_credentials(url: str, username: str, password: str) -> str:
    """Build the URL the decoder should dial.

    Credentials already in the URL are left alone — injecting a second copy is
    what turns a working address into a 401.
    """
    if not url or "://" not in url or not username:
        return url
    if has_credentials(url):
        return url
    scheme, rest = url.split("://", 1)
    user = quote(username, safe=_USERINFO_SAFE)
    secret = quote(password or "", safe=_USERINFO_SAFE)
    return f"{scheme}://{user}:{secret}@{rest}" if secret else f"{scheme}://{user}@{rest}"
