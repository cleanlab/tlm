"""Helper functions for parsing test values."""

import ast
import json
from typing import Any


def parse_dict_string(value: str) -> dict[str, Any]:
    """Parse a dict from a string representation."""
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        # Fall back to Python literal evaluation for single-quote strings
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, dict):
                return parsed
            raise ValueError(f"Parsed value is not a dict: {type(parsed)}")
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Could not parse dict string: {value}") from e
