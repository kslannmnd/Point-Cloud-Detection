from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf


def original_root() -> Path:
    try:
        return Path(get_original_cwd()).resolve()
    except ValueError:
        return Path.cwd().resolve()


def resolve_path(value: str | Path, root: Path | None = None) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (root or original_root()) / path


def project_path(cfg: DictConfig, value: str | Path) -> Path:
    root = resolve_path(cfg.project.root)
    return resolve_path(value, root=root)


def softgroup_repo_dir(cfg: DictConfig) -> Path:
    return project_path(cfg, cfg.softgroup.repo_dir)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_command(
    cmd: str | Sequence[str],
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    printable = cmd if isinstance(cmd, str) else " ".join(map(str, cmd))
    print(f"\n[RUN] {printable}\n")
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        env=dict(env) if env is not None else None,
        shell=isinstance(cmd, str),
        check=check,
    )


def repo_env(cfg: DictConfig, extra: Mapping[str, Any] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    repo_dir = softgroup_repo_dir(cfg)
    previous = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo_dir) + (os.pathsep + previous if previous else "")
    for key, value in cfg.softgroup.runtime_env.items():
        env[str(key)] = str(value)
    if extra:
        for key, value in extra.items():
            env[str(key)] = str(value)
    return env


def safe_remove_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    else:
        raise RuntimeError(f"Unsupported path type: {path}")


def ensure_symlink_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        print(f"Already exists: {dst}")
        return
    try:
        dst.symlink_to(src, target_is_directory=True)
        print(f"Symlink created: {dst} -> {src}")
    except OSError as exc:
        print(f"Symlink failed ({exc}); copying instead.")
        shutil.copytree(src, dst)


def copytree_only_rooms(src_root: Path, dst_root: Path, keep_rooms: Sequence[str]) -> list[str]:
    if not src_root.exists():
        raise FileNotFoundError(f"S3DIS source dataset not found: {src_root}")
    safe_remove_path(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    missing: list[str] = []
    for area_dir in sorted(p for p in src_root.iterdir() if p.is_dir() and p.name.startswith("Area_")):
        dst_area = dst_root / area_dir.name
        dst_area.mkdir(parents=True, exist_ok=True)
        for room_name in sorted(keep_rooms):
            src_room = area_dir / room_name
            dst_room = dst_area / room_name
            if src_room.is_dir():
                shutil.copytree(src_room, dst_room)
                copied.append(f"{area_dir.name}/{room_name}")
            else:
                missing.append(f"{area_dir.name}/{room_name}")

    print(f"Copied room folders: {len(copied)}")
    if missing:
        print(f"Missing requested room folders: {len(missing)}")
        for name in missing[:30]:
            print(f"  MISS {name}")
    return copied


def dump_effective_config(cfg: DictConfig, output_dir: Path) -> Path:
    ensure_dir(output_dir)
    path = output_dir / "effective_config.yaml"
    path.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    return path


def dump_json(data: Mapping[str, Any], output_path: Path) -> Path:
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def python_executable() -> str:
    return sys.executable
