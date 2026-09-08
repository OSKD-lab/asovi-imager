"""Dear PyGui dashboard for the ASoVi pipeline.

Importing this package does **not** import dearpygui — call :func:`launch`
(lazily) to start the GUI.  Launch it with::

    uv run python -m asvimg.gui [config.yaml | config_dir]
"""

from __future__ import annotations


def launch(config_path: str | None = None) -> None:
    """Start the dashboard (creates the single dearpygui context)."""
    from .app import run_app

    run_app(config_path)
