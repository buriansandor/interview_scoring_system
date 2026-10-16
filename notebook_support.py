"""Notebook wrappers around the same JSON loader and CLI used outside notebooks."""

import os
from pathlib import Path
import subprocess
import sys
import warnings

import pandas as pd

from llm_config import load_config, resolve_secondary_config


def _colab_secret(name):
    from google.colab import userdata
    return userdata.get(name)


def load_notebook_config(config_path, *, use_colab_secrets=False):
    """Use environment, then selected .key file, then optional selected Colab secret."""
    return load_config(
        config_path, require_file=True,
        secret_getter=_colab_secret if use_colab_secrets else None,
    )


def run_notebook(config_path, repo_dir, *, use_colab_secrets=False):
    """Run the real CLI and return its CSV; never print keys or change app.py."""
    config = load_notebook_config(config_path, use_colab_secrets=use_colab_secrets)
    repo_dir = Path(repo_dir).expanduser().resolve()
    app = repo_dir / "app.py"
    if not app.is_file():
        raise ValueError("repo_dir must point to the repository containing app.py.")
    environment = dict(os.environ)
    if config.api_key_env and config.llm.api_key:
        # The secret is passed privately to the child, never via arguments or disk.
        environment[config.api_key_env] = config.llm.api_key
    if config.secondary_llm is not None:
        try:
            secondary = resolve_secondary_config(
                config, secret_getter=_colab_secret if use_colab_secrets else None,
            )
        except (ValueError, OSError):
            warnings.warn(
                "Secondary credentials unavailable. The primary will still run; "
                "a needed second stage will be recorded as failed.",
                RuntimeWarning, stacklevel=2,
            )
        else:
            if secondary.api_key:
                environment[config.secondary_api_key_env] = secondary.api_key
    completed = subprocess.run(
        [sys.executable, str(app), "--config", str(config.config_path)],
        cwd=repo_dir, env=environment, check=False, capture_output=True, text=True,
    )
    if completed.returncode not in (0, 2):
        completed.check_returncode()
    if not config.output_csv.is_file():
        raise RuntimeError("The scoring process did not create the configured output CSV.")
    partial = completed.returncode == 2
    if partial:
        warnings.warn(
            "The shared request budget was exhausted. Returning saved partial results; no further calls.",
            RuntimeWarning, stacklevel=2,
        )
    result = pd.read_csv(config.output_csv)
    result.attrs["partial_results"] = partial
    result.attrs["provider"] = config.llm.provider
    result.attrs["model"] = config.llm.model
    result.attrs["output_csv"] = str(config.output_csv)
    return result
