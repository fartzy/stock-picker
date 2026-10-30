#!/usr/bin/env python3
"""
Why we need this shim:
- py_test requires a Python file as the main entry point
- We can't reference pytest's internal __main__.py directly as a Bazel label
- So we need our own Python file that calls pytest.main()
"""
import sys
from typing import NoReturn

import pytest


def main() -> NoReturn:
    """Forward Bazel test arguments and exit with pytest's original status code."""
    raise SystemExit(pytest.main(sys.argv[1:]))


if __name__ == "__main__":
    main()
