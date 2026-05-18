from __future__ import annotations

import re
import unicodedata


_METADATA_ALIASES = {
    "b": "bachelor",
    "ba": "bachelor",
    "bpo": "bachelor",
    "m": "master",
    "ma": "master",
    "mpo": "master",
    "sc": "science",
    "sciences": "science",
}


def normalize_metadata_value(value: object) -> str:
    normalized = unicodedata.normalize("NFKD", str(value).casefold())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = normalized.replace("&", "and").replace("/", " ")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return " ".join(_METADATA_ALIASES.get(token, token) for token in normalized.split())


def metadata_value_matches(actual: object, expected: object) -> bool:
    actual_norm = normalize_metadata_value(actual)
    expected_norm = normalize_metadata_value(expected)
    actual_tokens = set(actual_norm.split())
    expected_tokens = set(expected_norm.split())
    return (
        actual_norm == expected_norm
        or actual_norm in expected_norm
        or expected_norm in actual_norm
        or bool(actual_tokens and expected_tokens and actual_tokens.issubset(expected_tokens))
        or bool(actual_tokens and expected_tokens and expected_tokens.issubset(actual_tokens))
    )
