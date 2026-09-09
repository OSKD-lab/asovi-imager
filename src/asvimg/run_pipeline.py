"""Headless full-pipeline CLI (the ``asovi-run`` entry point).

Runs the entire ASoVi pipeline (preprocess → PCA/ICA → annotation → ROI →
correlation → export) end to end without any GUI, driving
:class:`~asvimg.runner.session.PipelineSession` with terminal
reporting.  Interactive steps are resolved non-interactively from the config
(``annotation`` = ``cache`` / coordinate pair / ``False``; ICA excludes
nothing).

Invoke it as a module (the ``asovi-run`` console script is only installed when
the project is built as a package — see pyproject's ``[project.scripts]``)::

    uv run python -m asvimg.run_pipeline --config <yaml|dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

# Headless: render figures off-screen so figure export never needs a display.
matplotlib.use("Agg")

from . import (  # noqa: E402  (after backend selection)
    PipelineConfig,
    TqdmReporter,
    default_output_dir,
    load_cli_config,
    load_config,
)
from .runner import PipelineSession  # noqa: E402

_STAGE_METHODS = {
    "preprocess": "run_preprocess",
    "pca": "run_pca",
    "ica": "run_ica",
    "annotation": "run_annotation",
    "roi": "run_roi",
    "correlation": "run_correlation",
    "export": "run_export",
}


def _resolve_output_dir(config: PipelineConfig) -> Path:
    if config.output_dir:
        return Path(config.output_dir)
    return default_output_dir(Path(config.input_dir), config.output_format)


def run_from_args(args: argparse.Namespace) -> None:
    config, config_source = load_cli_config(args.config)
    print(f"[asovi] {config_source}")
    if args.input_dir:
        config.input_dir = args.input_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.exp_name:
        config.exp_name = args.exp_name
    if args.output_format:
        config.output_format = args.output_format
    if args.max_frames is not None:
        config.max_frames = args.max_frames

    output_dir = _resolve_output_dir(config)
    figure_dir = None if args.no_figures else output_dir / "figures"
    reporter = TqdmReporter(figure_dir=figure_dir)

    session = PipelineSession(config, reporter=reporter)

    if args.stages:
        requested = [s.strip() for s in args.stages.split(",") if s.strip()]
        unknown = [s for s in requested if s not in _STAGE_METHODS]
        if unknown:
            raise SystemExit(
                f"Unknown stage(s): {unknown}. "
                f"Choose from {list(_STAGE_METHODS)}."
            )
        for stage in requested:
            getattr(session, _STAGE_METHODS[stage])()
    else:
        session.run_all()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full ASoVi pipeline headlessly "
            "(preprocess → PCA/ICA → annotation → ROI → correlation → export)."
        )
    )
    parser.add_argument(
        "--config",
        default=None,
        help="YAML config file or a directory with db.yaml/ops.yaml. "
             "Optional: without it, ./pipeline01_config.yaml is used when present, "
             "otherwise the built-in defaults",
    )
    parser.add_argument("--input-dir", default=None, help="Override input directory")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("--exp-name", default=None, help="Override experiment name")
    parser.add_argument(
        "--output-format", choices=["mat", "npy", "h5"], default=None,
        help="Override output format",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Limit frames for a quick test run",
    )
    parser.add_argument(
        "--stages", default=None,
        help=(
            "Comma-separated subset to run instead of all, e.g. "
            "'preprocess,pca'. Order is respected. Choices: "
            + ",".join(_STAGE_METHODS)
        ),
    )
    parser.add_argument(
        "--no-figures", action="store_true",
        help="Do not render/save QC figures (faster)",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_from_args(args)


if __name__ == "__main__":
    main()
