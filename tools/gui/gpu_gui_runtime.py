#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import time
from pathlib import Path


def project_root_path(current_file: Path) -> Path:
    current = current_file.resolve()
    for parent in [current.parent, *current.parents]:
        if (parent / 'CMakeLists.txt').exists() and (parent / 'src').exists():
            return parent
    return current.parent


def gpu_binary_path(project_root: Path) -> Path:
    return project_root / 'build' / 'sr_rt_gpu'


def outputs_dir(project_root: Path) -> Path:
    path = project_root / 'outputs'
    path.mkdir(parents=True, exist_ok=True)
    return path


def temp_output_base() -> Path:
    temp_dir = Path(tempfile.gettempdir()) / 'sr_rt_gpu_gui'
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir / f'gpu_gui_{time.time_ns()}'
