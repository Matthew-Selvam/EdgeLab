"""Allow ``python -m edgelab.evaluation`` to run the backtest pipeline."""

from __future__ import annotations

import sys

from .backtest import main

if __name__ == "__main__":
    sys.exit(main())
