"""
===============================================================================
Script Name   : logging_utils.py
Description   : Configures the Python logging facility for the project.
                Function: configure_logging
                - Sets up a dual-handler logger:
                  1. Console Handler: Prints info to the screen (stdout).
                  2. File Handler: Saves logs to 'logs/<dataset_name>/<run_name>.log'.
                - Ensures that previous handlers are cleared to prevent duplicate logs 
                  when running in interactive environments (like Jupyter/VS Code).

Usage:
    Called at the start of every script (training, attacks, preprocessing) 
    to initialize the logging system.

Author        : Dr. Samer Aoudi
Affiliation   : Higher Colleges of Technology (HCT), UAE
Role          : Assistant Professor & Division Chair (CIS)
Email         : cybersecurity@sameraoudi.com
ORCID         : 0000-0003-3887-0119
Created On    : 2025-Nov-22

License       : MIT License
Citation      : If this code is used in academic work, please cite the
                corresponding publication or acknowledge the author.
===============================================================================
"""

# src/utils/logging_utils.py
from __future__ import annotations

import logging
from pathlib import Path

from .paths import get_logs_root


def configure_logging(dataset_name: str, run_name: str = "default") -> None:
    """
    Configure root logger to log to console and to a file under logs/<dataset_name>/.

    Call this once at the start of each script / entry point.
    """
    logs_root = get_logs_root(dataset_name)
    logs_root.mkdir(parents=True, exist_ok=True)
    log_file = logs_root / f"{run_name}.log"

    # Basic formatter
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Clear existing handlers (avoid duplicate logs when re-running in notebooks)
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root_logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root_logger.addHandler(fh)

    root_logger.info("Logging configured. Writing to %s", log_file)
