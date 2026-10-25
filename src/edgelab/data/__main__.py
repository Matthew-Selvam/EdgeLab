"""Allow ``python -m edgelab.data`` to run the ingestion pipeline.

Usage::

    python -m edgelab.data
"""

from __future__ import annotations

import sys

from .ingest import main

if __name__ == "__main__":
    sys.exit(main())
