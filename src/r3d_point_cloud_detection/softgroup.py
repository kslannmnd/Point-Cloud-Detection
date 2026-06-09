from __future__ import annotations

from datetime import datetime
from pathlib import Path
import os
import re
import shutil
import site
import subprocess
import zipfile
from typing import Any

import yaml
from omegaconf import DictConfig, ListConfig, OmegaConf

from .utils import (
    copytree_only_rooms,
    dump_effective_config,
    dump_json,
    ensure_dir,
    ensure_symlink_or_copy,
    project_path,
    python_executable,
    repo_env,
    resolve_path,
    run_command,
    safe_remove_path,
    softgroup_repo_dir,
)


ALL_SPCONV_PACKAGES = [
    "spconv",
    "spconv-cu102",
    "spconv-cu111",
    "spconv-cu113",
    "spconv-cu114",
    "spconv-cu116",
    "spconv-cu117",
    "spconv-cu118",
    "spconv-cu120",
    "spconv-cu121",
    "spconv-cu124",
    "spconv-cu126",
    "spconv-cu128",
]
ALL_CUMM_PACKAGES = [
    "cumm",
    "cumm-cu102",
    "cumm-cu111",
    "cumm-cu113",
    "cumm-cu114",
    "cumm-cu116",
    "cumm-cu117",
    "cumm-cu118",
    "cumm-cu120",
    "cumm-cu121",
    "cumm-cu124",
    "cumm-cu126",
    "cumm-cu128",
]
SPCONV_CUDA_TAG_FALLBACKS = ["cu128", "cu126", "cu124", "cu121", "cu120", "cu118", "cu117", "cu116"]


def clone_softgroup(cfg: DictConfig) -> Path:
    repo_dir = softgroup_repo_dir(cfg)
    if repo_dir.exists():
        print(f"SoftGroup already exists: {repo_dir}")
        return repo_dir
    ensure_dir(repo_dir.parent)
    run_command(["git", "clone", str(cfg.softgroup.repo_url), str(repo_dir)])
    return repo_dir


def _cleanup_python_artifacts(pattern_names=("cumm", "spconv")) -> None:
    removed: list[str] = []
    for site_dir_raw in site.getsitepackages():
        site_dir = Path(site_dir_raw)
        if not site_dir.exists():
            continue
        for path in site_dir.iterdir():
            name = path.name.lower()
            if any(token in name for token in pattern_names):
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink(missing_ok=True)
                removed.append(str(path))
        pth = site_dir / "easy-install.pth"
        if pth.exists():
            text = pth.read_text(encoding="utf-8", errors="ignore")
            new_text = "\n".join(
                line for line in text.splitlines() if not any(token in line.lower() for token in pattern_names)
            )
            if new_text != text:
                pth.write_text(new_text, encoding="utf-8")
                removed.append(str(pth))
    print(f"Removed stale cumm/spconv artifacts: {len(removed)}")


def prepare_kaggle_env(cfg: DictConfig) -> None:
    if not cfg.env.is_kaggle:
        print("env.is_kaggle=false; skipping Kaggle-specific environment preparation.")
        return

    if cfg.env.use_apt and cfg.env.apt_packages:
        packages = " ".join(map(str, cfg.env.apt_packages))
        run_command(f"apt-get update && apt-get install -y {packages}", cwd="/")

    run_command([python_executable(), "-m", "pip", "install", "-U", "pip", "wheel", "setuptools"])
    run_command(
        [python_executable(), "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio", "functorch"],
        check=False,
    )
    run_command(
        [python_executable(), "-m", "pip", "uninstall", "-y", *ALL_CUMM_PACKAGES, *ALL_SPCONV_PACKAGES],
        check=False,
    )
    _cleanup_python_artifacts(("cumm", "spconv"))

    if cfg.env.kaggle.configure_api:
        run_command(["kaggle", "config", "view"], check=False)


def install_dependencies_and_build_softgroup(cfg: DictConfig) -> None:
    repo_dir = clone_softgroup(cfg)

    torch_cfg = cfg.env.torch
    if torch_cfg.target_torch:
        run_command(
            [
                python_executable(),
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "--force-reinstall",
                f"torch=={torch_cfg.target_torch}",
                f"torchvision=={torch_cfg.target_torchvision}",
                f"torchaudio=={torch_cfg.target_torchaudio}",
                "--index-url",
                str(torch_cfg.index_url),
            ]
        )

    run_command([python_executable(), "-m", "pip", "install", "--no-cache-dir", "-U", "numpy<2", "gdown"])
    softgroup_requirements = repo_dir / "requirements.txt"
    if softgroup_requirements.exists():
        _patch_softgroup_requirements_for_env(softgroup_requirements, cfg)
        run_command([python_executable(), "-m", "pip", "install", "--no-cache-dir", "-r", str(softgroup_requirements)])

    if cfg.env.spconv.get("cumm", "") or cfg.env.spconv.get("spconv", ""):
        run_command(
            [python_executable(), "-m", "pip", "uninstall", "-y", *ALL_CUMM_PACKAGES, *ALL_SPCONV_PACKAGES],
            check=False,
        )
        _cleanup_python_artifacts(("cumm", "spconv"))
    _install_spconv_stack(cfg)

    _patch_softgroup_ops_for_modern_torch(repo_dir)
    run_command([python_executable(), "setup.py", "build_ext", "--inplace"], cwd=repo_dir, env=repo_env(cfg))
    print("SoftGroup dependencies installed and native extension built.")


def _patch_softgroup_requirements_for_env(requirements_path: Path, cfg: DictConfig) -> None:
    blocked_prefixes: list[str] = []
    if str(cfg.env.get("name", "")) == "colab" or cfg.env.torch.get("target_torch", ""):
        blocked_prefixes.extend(["torch", "torchvision", "torchaudio"])
    if cfg.env.spconv.get("cumm", "") or cfg.env.spconv.get("spconv", ""):
        blocked_prefixes.extend(["cumm", "spconv"])
    if not blocked_prefixes:
        return

    lines = requirements_path.read_text(encoding="utf-8").splitlines()
    kept: list[str] = []
    removed: list[str] = []
    for line in lines:
        stripped = line.strip()
        package_name = re.split(r"[<>=!~\[]", stripped, maxsplit=1)[0].lower()
        if stripped and not stripped.startswith("#") and package_name in blocked_prefixes:
            removed.append(line)
        else:
            kept.append(line)
    if removed:
        requirements_path.write_text("\n".join(kept) + "\n", encoding="utf-8")
        print(f"Removed conflicting SoftGroup requirement lines: {removed}")


def _install_spconv_stack(cfg: DictConfig) -> None:
    cumm_package = str(cfg.env.spconv.get("cumm", "")).strip()
    spconv_package = str(cfg.env.spconv.get("spconv", "")).strip()
    if not cumm_package and not spconv_package:
        return
    if not cumm_package or not spconv_package:
        if cumm_package:
            run_command(
                [
                    python_executable(),
                    "-m",
                    "pip",
                    "install",
                    "--no-cache-dir",
                    "--force-reinstall",
                    cumm_package,
                ]
            )
        if spconv_package:
            run_command(
                [
                    python_executable(),
                    "-m",
                    "pip",
                    "install",
                    "--no-cache-dir",
                    "--force-reinstall",
                    spconv_package,
                ]
            )
        return

    candidates = _spconv_package_candidates(cumm_package, spconv_package)
    failures: list[str] = []
    for cumm_candidate, spconv_candidate, cuda_tag in candidates:
        try:
            print(f"Installing spconv stack: {cumm_candidate}, {spconv_candidate}")
            run_command(
                [
                    python_executable(),
                    "-m",
                    "pip",
                    "install",
                    "--no-cache-dir",
                    "--force-reinstall",
                    cumm_candidate,
                ]
            )
            run_command(
                [
                    python_executable(),
                    "-m",
                    "pip",
                    "install",
                    "--no-cache-dir",
                    "--force-reinstall",
                    "--no-deps",
                    spconv_candidate,
                ]
            )
            if cuda_tag:
                cfg.softgroup.runtime_env.CUMM_CUDA_VERSION = _cuda_version_from_tag(cuda_tag)
            print(f"Installed spconv stack: {cumm_candidate}, {spconv_candidate}")
            return
        except subprocess.CalledProcessError as exc:
            failures.append(f"{cumm_candidate}, {spconv_candidate}: exit {exc.returncode}")
            print(f"spconv stack candidate failed: {failures[-1]}")

    raise RuntimeError("Unable to install a compatible spconv stack. Tried: " + "; ".join(failures))


def _spconv_package_candidates(cumm_package: str, spconv_package: str) -> list[tuple[str, str, str | None]]:
    cumm_tag = _cuda_tag_from_package(cumm_package)
    spconv_tag = _cuda_tag_from_package(spconv_package)
    primary_tag = spconv_tag or cumm_tag
    if not primary_tag:
        return [(cumm_package, spconv_package, None)]

    tags = [primary_tag]
    tags.extend(tag for tag in SPCONV_CUDA_TAG_FALLBACKS if tag not in tags)
    return [
        (_package_for_cuda_tag(cumm_package, tag), _package_for_cuda_tag(spconv_package, tag), tag)
        for tag in tags
    ]


def _cuda_tag_from_package(package: str) -> str | None:
    match = re.search(r"-cu(\d+)$", package)
    return f"cu{match.group(1)}" if match else None


def _package_for_cuda_tag(package: str, cuda_tag: str) -> str:
    return re.sub(r"-cu\d+$", f"-{cuda_tag}", package)


def _cuda_version_from_tag(cuda_tag: str) -> str:
    digits = cuda_tag.removeprefix("cu")
    if len(digits) < 3:
        return digits
    return f"{digits[:-1]}.{digits[-1]}"


def _patch_softgroup_ops_for_modern_torch(repo_dir: Path) -> None:
    changed: list[str] = []
    for pattern in ("*.cpp", "*.cu", "*.h", "*.hpp"):
        for path in repo_dir.glob(f"softgroup/ops/**/*{pattern[1:]}"):
            src = path.read_text(encoding="utf-8", errors="ignore")
            patched = src.replace("AT_CHECK(", "TORCH_CHECK(").replace("AT_ASSERTM(", "TORCH_CHECK(")
            if patched != src:
                path.write_text(patched, encoding="utf-8")
                changed.append(str(path))
    if changed:
        print(f"Patched SoftGroup ops for modern PyTorch: {len(changed)} files")


def _s3dis_paths(cfg: DictConfig) -> tuple[Path, Path, Path]:
    repo_dataset_dir = project_path(cfg, cfg.dataset.repo_dataset_dir)
    raw_dir = repo_dataset_dir / str(cfg.dataset.raw_dir_name)
    src_dir = resolve_path(cfg.dataset.source_dir)
    return src_dir, repo_dataset_dir, raw_dir


def _patch_prepare_data_inst(repo_dataset_dir: Path) -> None:
    prepare_inst_path = repo_dataset_dir / "prepare_data_inst.py"
    if not prepare_inst_path.exists():
        print(f"No prepare_data_inst.py found at {prepare_inst_path}; skipping parser patch.")
        return

    src = prepare_inst_path.read_text(encoding="utf-8")
    patched = src
    room_repl = (
        "room_ver = pd.read_csv(raw_path, sep=r'\\s+', header=None, engine='python')"
        ".apply(pd.to_numeric, errors='coerce').dropna(axis=0, how='any').values"
    )

    patched = re.sub(
        r"room_ver\s*=\s*pd\.read_csv\(\s*raw_path\s*,\s*sep=['\"] ['\"]\s*,\s*header=None\s*\)\.values",
        lambda _: room_repl,
        patched,
    )
    obj_repl = (
        "obj_ver = pd.read_csv(single_object, sep=r'\\s+', header=None, engine='python')"
        ".apply(pd.to_numeric, errors='coerce').dropna(axis=0, how='any').values"
    )

    patched = re.sub(
        r"obj_ver\s*=\s*pd\.read_csv\(\s*single_object\s*,\s*sep=['\"] ['\"]\s*,\s*header=None\s*\)\.values",
        lambda _: obj_repl,
        patched,
    )
    patched = patched.replace(
        "rgb = np.ascontiguousarray(room_ver[:, 3:6], dtype='uint8')",
        "rgb = np.clip(np.rint(np.ascontiguousarray(room_ver[:, 3:6], dtype='float32')), 0, 255).astype('uint8')",
    )

    if patched != src:
        prepare_inst_path.write_text(patched, encoding="utf-8")
        print(f"Patched S3DIS parser: {prepare_inst_path}")
    else:
        print("S3DIS parser patch already applied or target code not found.")


def prepare_s3dis(cfg: DictConfig) -> None:
    clone_softgroup(cfg)
    src_dir, repo_dataset_dir, raw_dir = _s3dis_paths(cfg)
    ensure_dir(repo_dataset_dir)

    if cfg.dataset.get("preprocessed", {}).get("enabled", False):
        prepare_s3dis_preprocessed(cfg, repo_dataset_dir)
        return

    if cfg.dataset.crop.enabled:
        keep_rooms = list(cfg.dataset.crop.keep_rooms)
        print(f"Cropping S3DIS before SoftGroup preprocessing. keep_rooms={keep_rooms}")
        copytree_only_rooms(src_dir, raw_dir, keep_rooms)
    else:
        print("Using full S3DIS dataset before SoftGroup preprocessing.")
        safe_remove_path(raw_dir)
        ensure_symlink_or_copy(src_dir, raw_dir)

    _patch_prepare_data_inst(repo_dataset_dir)

    if not cfg.dataset.preprocess:
        print("dataset.preprocess=false; raw S3DIS prepared but prepare_data.sh was not run.")
        return

    preprocess_dir = repo_dataset_dir / "preprocess"
    preprocess_sample_dir = repo_dataset_dir / "preprocess_sample"
    val_gt_dir = repo_dataset_dir / "val_gt"
    if cfg.dataset.force_rebuild_preprocess:
        for path in (preprocess_dir, preprocess_sample_dir, val_gt_dir):
            if path.exists():
                shutil.rmtree(path)
                print(f"Removed stale preprocess directory: {path}")

    run_command(["bash", "prepare_data.sh"], cwd=repo_dataset_dir, env=repo_env(cfg))
    processed_files = sorted(preprocess_dir.glob("*_inst_nostuff.pth"))
    if not processed_files:
        raise RuntimeError(f"prepare_data.sh finished but {preprocess_dir} is empty.")
    print(f"S3DIS preprocess complete: {len(processed_files)} files.")


def prepare_s3dis_preprocessed(cfg: DictConfig, repo_dataset_dir: Path | None = None) -> None:
    clone_softgroup(cfg)
    repo_dataset_dir = repo_dataset_dir or project_path(cfg, cfg.dataset.repo_dataset_dir)
    ensure_dir(repo_dataset_dir)
    required_dirs = _preprocessed_required_dirs(cfg)
    source_dir = _resolve_preprocessed_source(cfg, required_dirs)

    for dir_name in required_dirs:
        src = source_dir / dir_name
        dst = repo_dataset_dir / str(dir_name)
        if not src.exists():
            raise FileNotFoundError(f"Required preprocessed S3DIS directory not found: {src}")
        safe_remove_path(dst)
        if str(cfg.dataset.preprocessed.copy_mode) == "copy":
            shutil.copytree(src, dst)
            print(f"Copied preprocessed S3DIS directory: {src} -> {dst}")
        else:
            ensure_symlink_or_copy(src, dst)

    processed_files = sorted((repo_dataset_dir / "preprocess").glob("*_inst_nostuff.pth"))
    if not processed_files:
        raise RuntimeError(f"Preprocessed S3DIS linked but preprocess/*.pth is empty: {repo_dataset_dir / 'preprocess'}")
    print(f"Preprocessed S3DIS is ready: {len(processed_files)} room files in {repo_dataset_dir / 'preprocess'}")


def _preprocessed_required_dirs(cfg: DictConfig) -> list[str]:
    return [str(item) for item in cfg.dataset.preprocessed.required_dirs]


def _find_preprocessed_root(path: Path, required_dirs: list[str]) -> Path | None:
    if not path.exists():
        return None
    candidates = [path]
    children = sorted([child for child in path.iterdir() if child.is_dir()], key=lambda item: _natural_key(item.name))
    candidates.extend(children)
    for child in children:
        candidates.extend(sorted([grandchild for grandchild in child.iterdir() if grandchild.is_dir()], key=lambda item: _natural_key(item.name)))
    for candidate in candidates:
        if all((candidate / dir_name).is_dir() for dir_name in required_dirs):
            return candidate
    return None


def _resolve_preprocessed_source(cfg: DictConfig, required_dirs: list[str]) -> Path:
    preprocessed_cfg = cfg.dataset.preprocessed
    source_dir = resolve_path(preprocessed_cfg.source_dir)
    source_root = _find_preprocessed_root(source_dir, required_dirs)
    if source_root:
        return source_root

    archive_value = preprocessed_cfg.get("archive_path")
    if archive_value:
        archive_path = resolve_path(archive_value)
        if archive_path.exists():
            extract_value = preprocessed_cfg.get("extract_dir")
            extract_dir = resolve_path(extract_value) if extract_value else archive_path.with_suffix("")
            extracted_root = _find_preprocessed_root(extract_dir, required_dirs)
            if extracted_root:
                return extracted_root
            ensure_dir(extract_dir)
            print(f"Extracting preprocessed S3DIS archive: {archive_path} -> {extract_dir}")
            with zipfile.ZipFile(archive_path, "r") as archive:
                archive.extractall(extract_dir)
            extracted_root = _find_preprocessed_root(extract_dir, required_dirs)
            if extracted_root:
                return extracted_root
            raise RuntimeError(f"Extracted archive does not contain required S3DIS directories: {required_dirs}")

    if source_dir.exists():
        raise FileNotFoundError(f"Preprocessed S3DIS source is missing required directories {required_dirs}: {source_dir}")
    raise FileNotFoundError(f"Preprocessed S3DIS source not found: {source_dir}")


def download_softgroup_checkpoints(cfg: DictConfig) -> dict[str, Path]:
    import gdown

    checkpoint_dir = project_path(cfg, cfg.softgroup.checkpoint_dir)
    ensure_dir(checkpoint_dir)
    downloaded: dict[str, Path] = {}
    for key, item in cfg.checkpoints.items():
        output_path = checkpoint_dir / str(item.name)
        if not output_path.exists():
            url = f"https://drive.google.com/uc?id={item.gdrive_id}"
            gdown.download(url=url, output=str(output_path), quiet=False)
        else:
            print(f"Checkpoint already exists: {output_path}")
        downloaded[str(key)] = output_path

    pretrained_target = checkpoint_dir / "softgroup_s3dis_pretrained.pth"
    source = downloaded.get("softgroup_s3dis")
    if source and source.exists() and not pretrained_target.exists():
        shutil.copy2(source, pretrained_target)
        print(f"Registered pretrained checkpoint: {pretrained_target}")
    return downloaded


def patch_softgroup_for_kaggle(cfg: DictConfig) -> None:
    repo_dir = clone_softgroup(cfg)
    _patch_softgroup_dataset_loaders_for_modern_torch(repo_dir)

    softgroup_model_path = repo_dir / "softgroup" / "model" / "softgroup.py"
    if softgroup_model_path.exists():
        src = softgroup_model_path.read_text(encoding="utf-8")
        patched = src.replace(
            "cur_proposals_idx = proposals_idx[mask_inds].long()",
            "cur_proposals_idx = proposals_idx.to(mask_inds.device)[mask_inds].long()",
        )
        if cfg.training.get("freeze_batch_norm", False) and "m.weight.requires_grad_(False)" not in patched:
            patched = re.sub(
                r"(?ms)^    def train\(self,\s*mode\s*=\s*True\):\n.*?(?=^    def |\Z)",
                """    def train(self, mode=True):
        super().train(mode)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.eval()
                if m.weight is not None:
                    m.weight.requires_grad_(False)
                if m.bias is not None:
                    m.bias.requires_grad_(False)

        return self

""",
                patched,
                count=1,
            )
        if patched != src:
            softgroup_model_path.write_text(patched, encoding="utf-8")
            print(f"Patched: {softgroup_model_path}")

    dist_util_path = repo_dir / "softgroup" / "util" / "dist.py"
    if dist_util_path.exists():
        src = dist_util_path.read_text(encoding="utf-8")
        old = "def collect_results_cpu(result_part, size, tmpdir=None):\n    rank, world_size = get_dist_info()"
        new = old + "\n    if world_size == 1:\n        return result_part"
        if old in src and "if world_size == 1:" not in src[src.find("def collect_results_cpu") : src.find("def collect_results_cpu") + 300]:
            dist_util_path.write_text(src.replace(old, new, 1), encoding="utf-8")
            print(f"Patched: {dist_util_path}")

    instance_eval_path = repo_dir / "softgroup" / "evaluation" / "instance_eval.py"
    if instance_eval_path.exists():
        src = instance_eval_path.read_text(encoding="utf-8")
        patched = re.sub(r"\bnp\.in1d\s*\(", "np.isin(", src)
        patched = re.sub(r"\bnp\.float\b", "float", patched)
        patched = re.sub(r"\bnp\.bool\b", "bool", patched)
        if patched != src:
            instance_eval_path.write_text(patched, encoding="utf-8")
            print(f"Patched: {instance_eval_path}")


def _patch_softgroup_dataset_loaders_for_modern_torch(repo_dir: Path) -> None:
    data_dir = repo_dir / "softgroup" / "data"
    if not data_dir.exists():
        return
    replacements = {
        "torch.load(filename)": "torch.load(filename, weights_only=False)",
        "torch.load(path)": "torch.load(path, weights_only=False)",
    }
    changed: list[Path] = []
    for path in sorted(data_dir.glob("*.py")):
        src = path.read_text(encoding="utf-8")
        patched = src
        for old, new in replacements.items():
            patched = patched.replace(old, new)
        if patched != src:
            path.write_text(patched, encoding="utf-8")
            changed.append(path)
    if changed:
        print(f"Patched SoftGroup dataset loaders for PyTorch torch.load defaults: {len(changed)} files")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _write_yaml(path: Path, data: dict[str, Any]) -> Path:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)
    return path


def _natural_key(value: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(value))]


def _plain_config(value):
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def discover_preprocessed_rooms(cfg: DictConfig) -> list[dict[str, str]]:
    _, repo_dataset_dir, _ = _s3dis_paths(cfg)
    preprocess_dir = repo_dataset_dir / "preprocess"
    rooms: list[dict[str, str]] = []
    for path in preprocess_dir.glob("*_inst_nostuff.pth"):
        match = re.match(r"^(Area_\d+)_(.+)_inst_nostuff\.pth$", path.name)
        if not match:
            continue
        area, room = match.group(1), match.group(2)
        rooms.append({"area": area, "room": room, "scene": f"{area}_{room}", "preprocess_path": str(path)})
    rooms = sorted(rooms, key=lambda row: (_natural_key(row["area"]), _natural_key(row["room"])))
    if not rooms:
        raise RuntimeError(f"No preprocessed S3DIS files found in {preprocess_dir}")
    return rooms


def discover_preprocessed_areas(cfg: DictConfig) -> list[str]:
    areas = sorted({row["area"] for row in discover_preprocessed_rooms(cfg)}, key=_natural_key)
    if not areas:
        _, repo_dataset_dir, _ = _s3dis_paths(cfg)
        raise RuntimeError(f"No preprocessed S3DIS files found in {repo_dataset_dir / 'preprocess'}")
    return areas


def choose_train_test_areas(cfg: DictConfig) -> tuple[list[str], str, str]:
    available_areas = discover_preprocessed_areas(cfg)
    preferred = str(cfg.dataset.fold.preferred_test_area)
    if preferred in available_areas:
        test_area = preferred
    elif len(available_areas) >= 2:
        test_area = available_areas[-1]
    else:
        test_area = available_areas[0]
    train_areas = [area for area in available_areas if area != test_area] or [test_area]
    area_tag = test_area.lower().replace("area_", "fold")
    return train_areas, test_area, area_tag


def _parse_room_item(item: Any) -> dict[str, str]:
    item = _plain_config(item)
    if isinstance(item, str):
        room_value = item.strip()
        if room_value.endswith("_inst_nostuff.pth"):
            room_value = room_value.removesuffix("_inst_nostuff.pth")
        if "/" in room_value:
            area, room = room_value.split("/", 1)
        elif ":" in room_value:
            area, room = room_value.split(":", 1)
        else:
            match = re.match(r"^(Area_\d+)_(.+)$", room_value)
            if not match:
                raise ValueError(f"Room string must look like Area_1/office_1, got {item!r}")
            area, room = match.group(1), match.group(2)
    elif isinstance(item, dict):
        area, room = item["area"], item["room"]
    else:
        raise TypeError(f"Unsupported S3DIS room selector: {item!r}")
    area = str(area).strip()
    room = str(room).strip()
    if not area or not room:
        raise ValueError(f"Invalid S3DIS room selector: {item!r}")
    return {"area": area, "room": room, "scene": f"{area}_{room}"}


def _explicit_room_items(value: Any, selected_rooms: Any) -> list[Any] | None:
    value = _plain_config(value)
    if isinstance(value, str):
        normalized = value.strip()
        if normalized in {"", "selected"}:
            return list(_plain_config(selected_rooms) or [])
        if normalized in {"all", "all_test_rooms", "all_trainable_rooms", "all_areas", "all_dataset"}:
            return None
        return [normalized]
    return list(value or [])


def resolve_s3dis_room_selection(
    cfg: DictConfig,
    rooms_value: Any,
    selected_rooms: Any,
    max_rooms: Any,
    default_scope: str,
    enforce_scope: bool,
) -> list[dict[str, str]]:
    all_rooms = discover_preprocessed_rooms(cfg)
    train_areas, test_area, _ = choose_train_test_areas(cfg)
    by_scene = {row["scene"]: row for row in all_rooms}
    plain_rooms_value = _plain_config(rooms_value)
    explicit_items = _explicit_room_items(plain_rooms_value, selected_rooms)

    if explicit_items is None:
        value = str(plain_rooms_value).strip()
        if value in {"all_test_rooms"} or (value == "all" and default_scope == "test"):
            rows = [row for row in all_rooms if row["area"] == test_area]
        elif value in {"all_trainable_rooms"} or (value == "all" and default_scope == "train"):
            rows = [row for row in all_rooms if row["area"] in train_areas]
        elif value in {"all_areas", "all_dataset"}:
            rows = list(all_rooms)
        else:
            rows = [row for row in all_rooms if row["area"] == test_area] if default_scope == "test" else [row for row in all_rooms if row["area"] in train_areas]
    else:
        parsed = [_parse_room_item(item) for item in explicit_items]
        missing = [row["scene"] for row in parsed if row["scene"] not in by_scene]
        if missing:
            raise ValueError(f"Requested S3DIS rooms are not present in preprocess: {missing}")
        rows = [by_scene[row["scene"]] for row in parsed]

    if enforce_scope and default_scope == "test":
        invalid = [row["scene"] for row in rows if row["area"] != test_area]
        if invalid:
            raise ValueError(f"Metric rooms must belong to all_test_rooms ({test_area}): {invalid}")
    if enforce_scope and default_scope == "train":
        invalid = [row["scene"] for row in rows if row["area"] not in train_areas]
        if invalid:
            raise ValueError(f"Training rooms must belong to all_trainable_rooms ({train_areas}): {invalid}")

    if max_rooms is not None:
        rows = rows[: int(max_rooms)]
    if not rows:
        raise ValueError("No S3DIS rooms selected.")
    return rows


def resolve_training_room_selection(cfg: DictConfig) -> list[dict[str, str]]:
    return resolve_s3dis_room_selection(
        cfg,
        rooms_value=cfg.training.get("rooms", "all_trainable_rooms"),
        selected_rooms=cfg.training.get("selected_rooms", []),
        max_rooms=cfg.training.get("max_rooms"),
        default_scope="train",
        enforce_scope=bool(cfg.training.get("enforce_trainable_areas", True)),
    )


def generate_softgroup_configs(cfg: DictConfig) -> dict[str, Path]:
    repo_dir = clone_softgroup(cfg)
    config_dir = project_path(cfg, cfg.softgroup.generated_config_dir)
    checkpoint_dir = project_path(cfg, cfg.softgroup.checkpoint_dir)
    train_areas, test_area, area_tag = choose_train_test_areas(cfg)
    available_areas = discover_preprocessed_areas(cfg)
    training_rooms = resolve_training_room_selection(cfg)

    official_backbone = repo_dir / "configs" / "softgroup" / "softgroup_s3dis_backbone_fold5.yaml"
    official_full = repo_dir / "configs" / "softgroup" / "softgroup_s3dis_fold5.yaml"
    backbone_cfg = _load_yaml(official_backbone)
    full_cfg = _load_yaml(official_full)

    work_dir = project_path(cfg, cfg.softgroup.work_dir)
    results_dir = project_path(cfg, cfg.inference.output_dir) / f"s3dis_{area_tag}"
    common_data_root = "dataset/s3dis/preprocess"
    train_prefix = _training_prefix(cfg, train_areas, training_rooms)

    infer_cfg = yaml.safe_load(yaml.safe_dump(full_cfg))
    infer_prefix = _inference_prefix(cfg, available_areas, test_area)
    infer_cfg["data"]["test"]["prefix"] = infer_prefix
    infer_cfg["data"]["test"]["data_root"] = common_data_root
    infer_cfg["dataloader"]["test"]["batch_size"] = int(cfg.training.get("test_batch_size", 1))
    infer_cfg["dataloader"]["test"]["num_workers"] = 1
    infer_cfg["work_dir"] = str(results_dir)

    backbone_train_cfg = yaml.safe_load(yaml.safe_dump(backbone_cfg))
    _apply_train_settings(
        backbone_train_cfg,
        train_prefix=train_prefix,
        test_area=test_area,
        data_root=common_data_root,
        lr=_stage_value(cfg.training.get("lr", 0.001), "backbone", 0.001, float),
        epochs=_stage_value(cfg.training.get("epochs", 2), "backbone", 2, int),
        train_batch_size=int(cfg.training.get("train_batch_size", 1)),
        test_batch_size=int(cfg.training.get("test_batch_size", 1)),
        num_workers=int(cfg.training.get("num_workers", 2)),
        work_dir=work_dir / f"{cfg.training.experiment_name}_backbone",
        pretrain=checkpoint_dir / str(cfg.checkpoints.hais_backbone.name),
    )
    _disable_x4_split(backbone_train_cfg)

    full_train_cfg = yaml.safe_load(yaml.safe_dump(full_cfg))
    _apply_train_settings(
        full_train_cfg,
        train_prefix=train_prefix,
        test_area=test_area,
        data_root=common_data_root,
        lr=_stage_value(cfg.training.get("lr", 0.001), "full_softgroup", 0.001, float),
        epochs=_stage_value(cfg.training.get("epochs", 2), "full_softgroup", 2, int),
        train_batch_size=int(cfg.training.get("train_batch_size", 1)),
        test_batch_size=int(cfg.training.get("test_batch_size", 1)),
        num_workers=int(cfg.training.get("num_workers", 2)),
        work_dir=work_dir / f"{cfg.training.experiment_name}_full_softgroup",
        pretrain=_full_softgroup_pretrain(cfg, work_dir),
    )

    paths = {
        "infer": _write_yaml(config_dir / f"softgroup_s3dis_{area_tag}_infer.yaml", infer_cfg),
        "backbone": _write_yaml(config_dir / f"softgroup_s3dis_{area_tag}_backbone_train.yaml", backbone_train_cfg),
        "full_softgroup": _write_yaml(config_dir / f"softgroup_s3dis_{area_tag}_full_train.yaml", full_train_cfg),
    }
    dump_json(
        {
            "available_train_areas": train_areas,
            "available_inference_areas": available_areas,
            "all_trainable_rooms": [row["scene"] for row in training_rooms],
            "test_area": test_area,
            "inference_prefix": infer_prefix,
            "area_tag": area_tag,
            "configs": {key: str(path) for key, path in paths.items()},
        },
        config_dir / "generation_metadata.json",
    )
    print("Generated SoftGroup configs:")
    for key, path in paths.items():
        print(f"  {key}: {path}")
    return paths


def _training_prefix(cfg: DictConfig, train_areas: list[str], training_rooms: list[dict[str, str]]):
    rooms_value = str(cfg.training.get("rooms", "all_trainable_rooms")).strip()
    if rooms_value in {"all", "all_trainable_rooms"} and cfg.training.get("max_rooms") is None:
        return train_areas
    return [row["scene"] for row in training_rooms]


def _full_softgroup_pretrain(cfg: DictConfig, work_dir: Path) -> Path:
    pretrain_checkpoint = cfg.training.get("pretrain_checkpoint")
    if pretrain_checkpoint:
        return project_path(cfg, pretrain_checkpoint)
    return work_dir / f"{cfg.training.experiment_name}_backbone" / "latest.pth"


def _inference_prefix(cfg: DictConfig, available_areas: list[str], default_area: str):
    if cfg.inference.mode != "s3dis":
        return default_area
    target = cfg.inference.target
    if target.kind in {"all", "all_test_rooms"}:
        rooms = resolve_s3dis_room_selection(
            cfg,
            rooms_value="all_test_rooms",
            selected_rooms=[],
            max_rooms=None,
            default_scope="test",
            enforce_scope=True,
        )
        return [row["scene"] for row in rooms]
    if target.kind == "all_areas":
        return available_areas
    if target.kind == "room":
        area = str(target.area)
        if area not in available_areas:
            raise ValueError(f"Requested inference area {area!r} is not preprocessed. Available: {available_areas}")
        return area
    return default_area


def _stage_value(value: Any, stage: str, default: Any, caster):
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        return caster(value.get(stage, default))
    return caster(value)


def _apply_train_settings(
    softgroup_cfg: dict[str, Any],
    train_prefix: list[str] | str,
    test_area: str,
    data_root: str,
    lr: float,
    epochs: int,
    train_batch_size: int,
    test_batch_size: int,
    num_workers: int,
    work_dir: Path,
    pretrain: Path,
) -> None:
    softgroup_cfg["data"]["train"]["prefix"] = train_prefix
    softgroup_cfg["data"]["train"]["data_root"] = data_root
    softgroup_cfg["data"]["test"]["prefix"] = test_area
    softgroup_cfg["data"]["test"]["data_root"] = data_root
    softgroup_cfg["optimizer"]["lr"] = lr
    softgroup_cfg["epochs"] = epochs
    softgroup_cfg["dataloader"]["train"]["batch_size"] = train_batch_size
    softgroup_cfg["dataloader"]["train"]["num_workers"] = num_workers
    softgroup_cfg["dataloader"]["test"]["batch_size"] = test_batch_size
    softgroup_cfg["dataloader"]["test"]["num_workers"] = 1
    softgroup_cfg["work_dir"] = str(work_dir)
    softgroup_cfg["pretrain"] = str(pretrain)

    softgroup_cfg["save_freq"] = 1


def _disable_x4_split(softgroup_cfg: dict[str, Any]) -> None:
    softgroup_cfg.setdefault("model", {}).setdefault("test_cfg", {})["x4_split"] = False
    softgroup_cfg.setdefault("data", {})
    for split in ("train", "test"):
        if isinstance(softgroup_cfg["data"].get(split), dict) and "x4_split" in softgroup_cfg["data"][split]:
            softgroup_cfg["data"][split]["x4_split"] = False


def train_softgroup(cfg: DictConfig) -> list[Path]:
    if cfg.training.mode == "none_download":
        downloaded = download_softgroup_checkpoints(cfg)
        source = downloaded[str(cfg.training.checkpoint_key)]
        target = project_path(cfg, cfg.training.save_dir) / str(cfg.training.checkpoint_name)
        if not target.exists():
            shutil.copy2(source, target)
        print(f"Training skipped. Registered checkpoint: {target}")
        return [target]

    if cfg.training.mode == "external_train_file":
        output_dir = ensure_dir(project_path(cfg, cfg.softgroup.work_dir) / str(cfg.training.experiment_name))
        effective_config = dump_effective_config(cfg, output_dir)
        train_file = project_path(cfg, cfg.training.train_file)
        if not train_file.exists():
            raise FileNotFoundError(f"External train file not found: {train_file}")
        run_command([python_executable(), str(train_file), str(effective_config)], env=repo_env(cfg))
        return []

    patch_softgroup_for_kaggle(cfg)
    config_paths = generate_softgroup_configs(cfg)
    repo_dir = softgroup_repo_dir(cfg)
    registered: list[Path] = []

    for stage in cfg.training.stages:
        stage = str(stage)
        if stage not in config_paths:
            raise ValueError(f"No generated config for training stage {stage!r}")
        cfg_path = config_paths[stage]
        train_cfg = _load_yaml(cfg_path)
        work_dir = Path(train_cfg["work_dir"])
        if stage == "full_softgroup":
            pretrain = Path(train_cfg["pretrain"])
            if not pretrain.exists():
                raise FileNotFoundError(f"Full SoftGroup training needs backbone checkpoint first: {pretrain}")
        run_command([python_executable(), "tools/train.py", str(cfg_path)], cwd=repo_dir, env=repo_env(cfg))
        latest = work_dir / "latest.pth"
        if latest.exists():
            registered.append(register_checkpoint(cfg, latest, f"{cfg.training.experiment_name}_{stage}_latest.pth"))
    return registered


def register_checkpoint(cfg: DictConfig, source: Path, name: str) -> Path:
    target = project_path(cfg, cfg.training.save_dir) / name
    ensure_dir(target.parent)
    shutil.copy2(source, target)
    print(f"Registered checkpoint: {target}")
    return target


def run_s3dis_inference(cfg: DictConfig) -> Path:
    if cfg.inference.target.kind == "room":
        return run_s3dis_room_inference(cfg)

    config_paths = generate_softgroup_configs(cfg)
    softgroup_cfg_path = config_paths["infer"]
    checkpoint = project_path(cfg, cfg.inference.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Inference checkpoint not found: {checkpoint}")

    target = cfg.inference.target
    output_root = project_path(cfg, cfg.inference.output_dir)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_name = str(target.kind)
    if target.kind == "room":
        target_name = f"{target.area}_{target.room}"
    run_dir = ensure_dir(output_root / f"{run_id}_{target_name}")

    run_command(
        [python_executable(), "tools/test.py", str(softgroup_cfg_path), str(checkpoint), "--out", str(run_dir / "raw")],
        cwd=softgroup_repo_dir(cfg),
        env=repo_env(cfg),
    )

    effective_config = dump_effective_config(cfg, run_dir)
    dump_json(
        {
            "checkpoint_path": str(checkpoint),
            "config_path": str(effective_config),
            "softgroup_config_path": str(softgroup_cfg_path),
            "target": OmegaConf.to_container(target, resolve=True),
            "raw_output_dir": str(run_dir / "raw"),
        },
        run_dir / "provenance.json",
    )
    print(f"Inference output saved to: {run_dir}")
    return run_dir


def run_s3dis_room_inference(cfg: DictConfig) -> Path:
    from .s3dis_visualization import (
        apply_visualization_defaults,
        context_from_cfg,
        save_room_bbox_csv,
        show_room_bbox,
        soft_group_get_instance_seg,
    )

    target = cfg.inference.target
    output_root = project_path(cfg, cfg.inference.output_dir)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    scene = f"{target.area}_{target.room}"
    run_dir = ensure_dir(output_root / f"{run_id}_{scene}")
    context = context_from_cfg(cfg)
    context.results_root = ensure_dir(run_dir / "raw")
    context.html_dir = ensure_dir(run_dir / "html")
    context.checkpoint_path = project_path(cfg, cfg.inference.checkpoint)
    context.save_html = bool(cfg.inference.save_visualization)
    context.show_inline = bool(cfg.visualization.show_inline)

    inference_data = soft_group_get_instance_seg(
        target.area,
        target.room,
        context=context,
        force_run=True,
        display_tables=bool(cfg.visualization.display_tables),
        allow_precomputed_results=False,
    )
    _, bbox_df = show_room_bbox(
        target.area,
        target.room,
        context=context,
        inference_data=inference_data,
        **apply_visualization_defaults(cfg, "bbox"),
    )
    bbox_csv = save_room_bbox_csv(bbox_df, target.area, target.room, context=context, out_dir=run_dir) if len(bbox_df) else None
    effective_config = dump_effective_config(cfg, run_dir)
    dump_json(
        {
            "checkpoint_path": str(context.checkpoint_path),
            "config_path": str(effective_config),
            "target": OmegaConf.to_container(target, resolve=True),
            "raw_output_dir": str(context.results_root),
            "html_dir": str(context.html_dir),
            "bbox_csv": str(bbox_csv) if bbox_csv else None,
        },
        run_dir / "provenance.json",
    )
    print(f"S3DIS room inference output saved to: {run_dir}")
    return run_dir
