"""``python -m asvimg.gui [config]`` entry point."""

from __future__ import annotations

import sys

from . import launch

if __name__ == "__main__":
    launch(sys.argv[1] if len(sys.argv) > 1 else None)
