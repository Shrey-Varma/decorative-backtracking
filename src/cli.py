"""Shared CLI plumbing for the numbered scripts: config + overrides + run dir
+ manifest, so every stage records exactly what produced its outputs (§4)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import Config, load_config
from .io_utils import resolve_run_dir, write_manifest


def script_setup(description: str, default_config: str = "configs/default.yaml"):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=default_config)
    parser.add_argument(
        "--run-dir", default=None,
        help="shared output dir across stages; created by 01, required after",
    )
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[],
        metavar="KEY.PATH=VALUE", help="config override (repeatable)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg: Config = load_config(args.config, args.overrides)
    cfg_dict = cfg.model_dump()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = resolve_run_dir(cfg_dict, cfg.run_name, cfg.experiment.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(run_dir / "manifests" / description.split()[0], cfg_dict)
    return cfg, cfg_dict, run_dir
