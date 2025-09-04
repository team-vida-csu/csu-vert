# src/vert/weights.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import hashlib
import json
import os
import sys
import time
import requests

# -----------------------------
# Registry (GitHub Releases only)
# -----------------------------

@dataclass(frozen=True)
class ModelEntry:
    format: str              # "pt" or "onnx"
    weight_name: str         # e.g., "vert-unet_resnet34_4cls_scripted.pt"
    yaml_name: str           # e.g., "vert-model.yaml"
    gh_owner: str            # GitHub owner/org
    gh_repo: str             # GitHub repo name
    sha256: Optional[Dict[str, str]] = None  # filename -> sha256

# key = (model_id, version)
MODELS: Dict[Tuple[str, str], ModelEntry] = {
    ("unet-r34-4c-pt",   "v0.0.2"): ModelEntry(
        format="pt",
        weight_name="vert-unet_resnet34_4cls_scripted.pt",
        yaml_name="vert-model.yaml",
        gh_owner="team-vida-csu",
        gh_repo="csu-vert",
        sha256={
            "vert-unet_resnet34_4cls_scripted.pt": "efd642f25b554e101d26c6c2ba7c2b02870eee896fb47147c2bfe4511ea7c3cc",
            "vert-model.yaml": "09affdc7b905f21b255f5bc7c14ea5e7e55984de11b5f88e195162b19ba29865",
        },
    ),
    ("unet-r34-4c-onnx", "v0.0.2"): ModelEntry(
        format="onnx",
        weight_name="vert-unet_resnet34_4cls.onnx",
        yaml_name="vert-model.yaml",
        gh_owner="team-vida-csu",
        gh_repo="csu-vert",
        sha256={
            "vert-unet_resnet34_4cls.onnx": "0b1951a6c6989fd88b847cbeccb53b05c80469d5e603a48169fcc1f201f0eaae",
            "vert-model.yaml": "09affdc7b905f21b255f5bc7c14ea5e7e55984de11b5f88e195162b19ba29865",
        },
    ),
}

# -----------------------------
# Public API
# -----------------------------

def resolve_weights(
    spec: str = "auto:unet-r34-4c@v0.0.2",
    preferred_format: str = "auto",  # "auto" | "pt" | "onnx"
) -> Tuple[Path, Optional[Path], Dict, str]:
    """
    Resolve weights to local cached files.

    spec forms:
      • Local path: /path/to/model.pt|onnx
      • auto:<model_id>@<version>   (uses registry; respects preferred_format)
      • gh:<owner>/<repo>@<tag>/<filename>  (explicit asset on GitHub Releases)

    Returns:
      (weights_path, yaml_path_or_None, yaml_dict_or_empty, format_str)
    """
    # Local file?
    p = Path(spec)
    if p.exists() and p.is_file():
        return p, None, {}, _infer_format(p.suffix)

    # auto:<id>@<ver>
    if spec.startswith("auto:"):
        ident = spec[5:]
        model_id, version = _split_id_version(ident)
        return _resolve_auto(model_id, version, preferred_format)

    # explicit GitHub release asset
    if spec.startswith("gh:"):
        owner, repo, tag, fname = _parse_gh_spec(spec)
        return _resolve_gh_explicit(owner, repo, tag, fname)

    # bare id@ver
    if "@" in spec:
        model_id, version = spec.split("@", 1)
        return _resolve_auto(model_id, version, preferred_format)

    raise ValueError(f"Unrecognized weights spec: {spec}")


def list_models() -> List[str]:
    """Human-readable registry summary."""
    out = []
    for (mid, ver), e in sorted(MODELS.items()):
        out.append(f"{mid}@{ver}  [{e.format}]  {e.weight_name}")
    return out


def precache(spec: str, preferred_format: str = "auto") -> Tuple[Path, Optional[Path]]:
    """Download/cache files for a spec, then return paths."""
    w, y, _, _ = resolve_weights(spec, preferred_format=preferred_format)
    return w, y

# -----------------------------
# Auto resolver (GitHub Release)
# -----------------------------

def _resolve_auto(
    model_id: str,
    version: str,
    preferred_format: str,
) -> Tuple[Path, Optional[Path], Dict, str]:
    key_pt = (model_id + "-pt", version)
    key_ox = (model_id + "-onnx", version)

    # choose entry
    if preferred_format == "pt" and key_pt in MODELS:
        key = key_pt
    elif preferred_format == "onnx" and key_ox in MODELS:
        key = key_ox
    else:
        key = key_pt if key_pt in MODELS else key_ox if key_ox in MODELS else None

    if key is None:
        raise KeyError(f"No registry entry for {model_id}@{version} (format={preferred_format}).")

    entry = MODELS[key]
    cdir = _cache_dir(model_id, version, entry.format)
    weight_p = cdir / entry.weight_name
    yaml_p = cdir / entry.yaml_name if entry.yaml_name else None

    # download if missing
    if not weight_p.exists() or (yaml_p and not yaml_p.exists()):
        base = f"https://github.com/{entry.gh_owner}/{entry.gh_repo}/releases/download/{version}/"
        _download_with_retries(base + entry.weight_name, weight_p)
        if yaml_p:
            _download_with_retries(base + entry.yaml_name, yaml_p)

    # verify
    if entry.sha256:
        _verify(weight_p, entry.sha256.get(entry.weight_name))
        if yaml_p:
            _verify(yaml_p, entry.sha256.get(entry.yaml_name))

    cfg = _parse_yaml(yaml_p) if (yaml_p and yaml_p.exists()) else {}
    return weight_p, yaml_p, cfg, entry.format


def _resolve_gh_explicit(owner: str, repo: str, tag: str, filename: str) -> Tuple[Path, Optional[Path], Dict, str]:
    fmt = _infer_format(Path(filename).suffix)
    cdir = _cache_dir(f"{owner}-{repo}", tag, fmt)
    dest = cdir / Path(filename).name
    if not dest.exists():
        url = f"https://github.com/{owner}/{repo}/releases/download/{tag}/{filename}"
        _download_with_retries(url, dest)
    # attempt to use a sibling YAML if present
    cfg = {}
    yaml_p = None
    for cand in ("vert-model.yaml", "model.yaml", "config.yaml"):
        yp = cdir / cand
        if yp.exists():
            yaml_p = yp
            cfg = _parse_yaml(yp) or {}
            break
    return dest, yaml_p, cfg, fmt

# -----------------------------
# Cache & helpers
# -----------------------------

def _cache_root() -> Path:
    root = os.environ.get("VERT_CACHE_DIR")
    if root:
        return Path(root)
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "vert"
    return Path.home() / ".cache" / "vert"

def _cache_dir(model_id: str, version: str, fmt: str) -> Path:
    p = _cache_root() / model_id / version / fmt
    p.mkdir(parents=True, exist_ok=True)
    return p

def _download_with_retries(url: str, dest: Path, retries: int = 3, backoff: float = 1.5) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            _download_stream(url, tmp)
            tmp.replace(dest)
            return
        except Exception as e:
            last_err = e
            if attempt == retries:
                raise
            time.sleep(backoff ** attempt)
    if last_err:
        raise last_err

def _download_stream(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        done = 0
        with open(dest, "wb") as f:
            for b in r.iter_content(chunk):
                if not b:
                    continue
                f.write(b)
                done += len(b)
                if total and sys.stderr.isatty():
                    pct = done * 100 // total
                    sys.stderr.write(f"\rDownloading {dest.name}: {pct}%")
                    sys.stderr.flush()
        if sys.stderr.isatty():
            sys.stderr.write("\n")

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _verify(path: Path, want: Optional[str]) -> None:
    if not want:
        return
    got = _sha256(path)
    if got.lower() != want.lower():
        raise ValueError(f"Checksum mismatch for {path.name}: got {got}, want {want}")

def _parse_yaml(path: Optional[Path]) -> Dict:
    if not path or not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="ignore")
    # try YAML, then JSON
    try:
        import yaml
        data = yaml.safe_load(text) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

def _split_id_version(ident: str) -> Tuple[str, str]:
    if "@" not in ident:
        raise ValueError("Use 'auto:<model_id>@<version>'")
    mid, ver = ident.split("@", 1)
    if not mid or not ver:
        raise ValueError("Use 'auto:<model_id>@<version>'")
    return mid, ver

def _parse_gh_spec(spec: str) -> Tuple[str, str, str, str]:
    # gh:owner/repo@tag/filename
    ident = spec[3:]
    if "@" not in ident or "/" not in ident:
        raise ValueError("Use 'gh:<owner>/<repo>@<tag>/<filename>'")
    owner_repo, _, rest = ident.partition("@")
    owner, repo = owner_repo.split("/", 1)
    if "/" not in rest:
        raise ValueError("Provide a filename after tag: gh:owner/repo@tag/file")
    tag, _, fname = rest.partition("/")
    return owner, repo, tag, fname

def _infer_format(suffix: str) -> str:
    return "onnx" if suffix.lower() == ".onnx" else "pt"
