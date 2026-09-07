import ipaddress
import re
from urllib.parse import urlparse


SUSPICIOUS_TOKENS = {
    "login",
    "signin",
    "sign-in",
    "verify",
    "verification",
    "secure",
    "account",
    "update",
    "confirm",
    "password",
    "credential",
    "bank",
    "payment",
    "billing",
    "invoice",
    "admin",
    "webmail",
    "auth",
    "unlock",
}


def _safe_parse(url: str):
    url = str(url or "").strip().replace("[.]", ".")

    if not re.match(r"^[a-z][a-z0-9+.-]*://", url, re.IGNORECASE):
        url = "http://" + url

    return urlparse(url)


def _has_valid_port(parsed) -> bool:
    try:
        return parsed.port is not None
    except ValueError:
        return False


def _is_ip_hostname(hostname: str) -> bool:
    if not hostname:
        return False

    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        return False


def extract_url_features(url: str) -> dict:
    raw_url = str(url or "").strip()
    parsed = _safe_parse(raw_url)

    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    labels = hostname.split(".") if hostname else []
    subdomain_count = 0 if _is_ip_hostname(hostname) else max(len(labels) - 2, 0)

    suspicious_token_count = sum(
        1
        for token in SUSPICIOUS_TOKENS
        if token in raw_url.lower()
    )

    return {
        "url_length": len(raw_url),
        "hostname_length": len(hostname),
        "path_length": len(path),
        "query_length": len(query),
        "dot_count": raw_url.count("."),
        "hyphen_count": raw_url.count("-"),
        "underscore_count": raw_url.count("_"),
        "slash_count": raw_url.count("/"),
        "question_mark_count": raw_url.count("?"),
        "equals_count": raw_url.count("="),
        "ampersand_count": raw_url.count("&"),
        "percent_count": raw_url.count("%"),
        "at_count": raw_url.count("@"),
        "digit_count": sum(c.isdigit() for c in raw_url),
        "special_char_count": sum(
            not c.isalnum() for c in raw_url
        ),
        "subdomain_count": subdomain_count,
        "has_ip_hostname": int(_is_ip_hostname(hostname)),
        "has_port": int(_has_valid_port(parsed)),
        "has_query": int(bool(query)),
        "has_fragment": int(bool(parsed.fragment)),
        "has_encoded_chars": int("%" in raw_url),
        "suspicious_token_count": suspicious_token_count,
    }


if __name__ == "__main__":
    examples = [
        "google.com",
        "https://example.com/login",
        "192.168.1.10/admin.php?verify=1",
    ]

    for example in examples:
        print()
        print("URL:", example)
        print(extract_url_features(example))
