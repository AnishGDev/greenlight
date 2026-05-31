"""Compatibility wrapper for the canonical GreenLight schemas package.

The runtime import path is the `schemas/` package. Keep this module as a thin
re-export so direct file references do not drift into a second schema contract.
"""

from schemas.schemas import *  # noqa: F401,F403
