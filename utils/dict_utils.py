"""Shared dictionary utilities."""
from __future__ import annotations

import copy
from typing import Any, Dict


def deep_merge(
    base: Dict[str, Any],
    override: Dict[str, Any],
    *,
    deep_copy: bool = False,
    skip_private_keys: bool = False,
) -> Dict[str, Any]:
    """
    Deep merge two dictionaries. Override values take precedence.

    - For dicts: recursively merge
    - For lists / scalars: override replaces base

    Args:
        base: Base dictionary.
        override: Dictionary whose values take precedence.
        deep_copy: When *True*, values are ``copy.deepcopy``'d so the
            caller's original dicts are never mutated.  When *False* a
            shallow ``.copy()`` is used (faster, fine for read-only usage).
        skip_private_keys: When *True*, keys starting with ``_`` in
            *override* are silently skipped (useful for meta-keys like
            ``_inherits``).
    """
    if not override:
        return base

    result = copy.deepcopy(base) if deep_copy else base.copy()

    for key, value in override.items():
        if skip_private_keys and key.startswith("_"):
            continue

        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(
                result[key],
                value,
                deep_copy=deep_copy,
                skip_private_keys=skip_private_keys,
            )
        else:
            result[key] = copy.deepcopy(value) if deep_copy else value

    return result
