"""Utility functions to generate basic JSON payloads for benchmarking.

This module exposes helper functions to create deterministic payloads of
varying shapes and approximate sizes. The payloads are limited to JSON
native types so they are eligible for the Rust fast path in DataSON.
"""

from __future__ import annotations

import json
import random
from typing import Any, Tuple

RANDOM_SEED = 1234


def _random_string(length: int) -> str:
    random.seed(RANDOM_SEED)
    letters = "abcdefghijklmnopqrstuvwxyz"
    return "".join(random.choice(letters) for _ in range(length))


def make_flat(n_bytes: int) -> Tuple[Any, str]:
    """Return a flat list of dictionaries roughly ``n_bytes`` in size.

    The function returns both the Python object and its JSON string
    representation so that benchmarks can test ``save_string`` and
    ``load_basic`` separately.
    """
    item = {"id": 1, "text": _random_string(10)}
    data = [item] * max(n_bytes // 20, 1)
    json_data = json.dumps(data)
    return data, json_data


def make_nested(n_bytes: int) -> Tuple[Any, str]:
    """Return a nested dictionary/list structure around ``n_bytes``."""
    data = {"level1": {"level2": [i for i in range(max(n_bytes // 10, 1))]}}
    json_data = json.dumps(data)
    return data, json_data


def make_mixed(n_bytes: int) -> Tuple[Any, str]:
    """Return a structure that mixes lists and dicts.

    The resulting payload still uses only JSON native types but has a
    slightly irregular structure compared to :func:`make_flat`.
    """
    data = [{"items": [j for j in range(3)]} for _ in range(max(n_bytes // 50, 1))]
    json_data = json.dumps(data)
    return data, json_data

