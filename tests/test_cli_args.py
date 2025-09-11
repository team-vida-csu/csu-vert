from pathlib import Path
from typing import Any, Dict, List, Tuple
import pytest

# Import the module that defines main()/build_parser()
import vert.cli as cli


class _Recorder:
    """Capture calls to infer.run_folder and weights funcs."""
    def __init__(self):
        self.calls: List[Tuple[str, Dict[str, Any]]] = []

    def __call__(self, **kwargs):
        self.calls.append(("run_folder", kwargs))


@pytest.fixture
def mock_infer(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr("vert.cli.infer.run_folder", rec, raising=True)
    return rec

@pytest.fixture
def mock_weights(monkeypatch, tmp_path: Path):
    """
    Stub resolve_weights/list_models/precache so tests are fast/offline.
    """
    weights_path = tmp_path / "dummy.pt"
    weights_path.write_bytes(b"pt")  # exists so str(path) is valid
    cfg = {
        "normalize_mean": [0.1, 0.2, 0.3],
        "normalize_std": [0.9, 0.8, 0.7],
        "class_names": ["background", "forb", "graminoid", "woody"],
        "num_classes": 4,
        "palette": [(0,0,0),(1,2,3),(4,5,6),(7,8,9)],
    }

    def resolve_weights(spec: str, preferred_format: str = "auto"):
        # Return: (weights_path, yaml_path_or_None, cfg_dict, format_str)
        return weights_path, tmp_path / "cfg.yaml", cfg, "pt"

    def list_models():
        return ["unet-r34-4c@v0.0.2  [pt]  vert-unet_resnet34_4cls_scripted.pt"]

    def precache(spec: str, preferred_format: str = "auto"):
        # mimic download result
        return weights_path, tmp_path / "cfg.yaml"

    monkeypatch.setattr("vert.cli.resolve_weights", resolve_weights, raising=True)
    monkeypatch.setattr("vert.cli.list_models", list_models, raising=True)
    monkeypatch.setattr("vert.cli.precache", precache, raising=True)
    return {"weights_path": weights_path, "cfg": cfg}


def run_cli(argv: list[str]) -> int:
    return cli.main(argv)


def test_help_prints_ok(capsys):
    # argparse handles --help internally; ensure it prints and exits(0)
    with pytest.raises(SystemExit) as se:
        cli.build_parser().parse_args(["infer", "--help"])
    assert se.value.code == 0
    out = capsys.readouterr().out
    assert "--input" in out and "--weights" in out


def test_infer_required_args_pass(mock_infer, mock_weights, tmp_path):
    imgs = tmp_path / "imgs"
    out = tmp_path / "out"
    imgs.mkdir()
    out.mkdir()
    # Call with only required args + simplest options
    rc = run_cli([
        "infer",
        "--input", str(imgs),
        "--output", str(out),
        "--weights", "auto:unet-r34-4c@v0.0.2",
    ])
    assert rc == 0
    assert mock_infer.calls, "infer.run_folder should be called once"
    _, kwargs = mock_infer.calls[-1]
    # spot-check a few defaults wired through
    assert kwargs["tile_size"] == 512
    assert kwargs["overlap"] == 64
    assert kwargs["batch_size"] == 4
    assert kwargs["save_side_by_side"] is True
    assert kwargs["save_overlay"] is False
    # YAML defaults merged:
    assert tuple(kwargs["mean"]) == (0.1, 0.2, 0.3)
    assert tuple(kwargs["std"]) == (0.9, 0.8, 0.7)
    assert kwargs["class_names"] == ["background", "forb", "graminoid", "woody"]


@pytest.mark.parametrize("flag,expect_key,expect_val", [
    ("--overlay",          "save_overlay", True),
    ("--amp",              "amp", True),
    ("--suppress-noise",   "suppress_noise", True),
])
def test_boolean_flags_true(mock_infer, mock_weights, tmp_path, flag, expect_key, expect_val):
    imgs = (tmp_path / "imgs") 
    imgs.mkdir()
    out = (tmp_path / "out")
    out.mkdir()
    rc = run_cli([
        "infer", "-i", str(imgs), "-o", str(out),
        "-w", "auto:unet-r34-4c@v0.0.2",
        flag,
    ])
    assert rc == 0
    _, kwargs = mock_infer.calls[-1]
    assert kwargs[expect_key] is expect_val


def test_side_by_side_toggle(mock_infer, mock_weights, tmp_path):
    imgs = (tmp_path / "imgs")
    imgs.mkdir()
    out = (tmp_path / "out")
    out.mkdir()
    rc = run_cli([
        "infer", "-i", str(imgs), "-o", str(out),
        "-w", "auto:unet-r34-4c@v0.0.2",
        "--no-side-by-side",
    ])
    assert rc == 0
    _, kwargs = mock_infer.calls[-1]
    assert kwargs["save_side_by_side"] is False


def test_numeric_args_and_lists_override(mock_infer, mock_weights, tmp_path):
    imgs = (tmp_path / "imgs")
    imgs.mkdir()
    out = (tmp_path / "out")
    out.mkdir()
    rc = run_cli([
        "infer",
        "-i", str(imgs),
        "-o", str(out),
        "-w", "auto:unet-r34-4c@v0.0.2",
        "--tile-size", "768",
        "--overlap", "128",
        "--batch-size", "2",
        "--overlay-alpha", "200",
        "--mean", "0.5", "0.4", "0.3",
        "--std", "0.2", "0.2", "0.2",
        "--class-names", "bg,aa,bb,cc",
        "--csv", str(out / "stats.csv"),
        "--device", "cpu",
        "--format", "pt",
    ])
    assert rc == 0
    _, kwargs = mock_infer.calls[-1]
    assert kwargs["tile_size"] == 768
    assert kwargs["overlap"] == 128
    assert kwargs["batch_size"] == 2
    assert kwargs["overlay_alpha"] == 200
    assert tuple(kwargs["mean"]) == (0.5, 0.4, 0.3)
    assert tuple(kwargs["std"]) == (0.2, 0.2, 0.2)
    assert kwargs["class_names"] == ["bg", "aa", "bb", "cc"]
    assert kwargs["csv_path"].endswith("stats.csv")
    assert kwargs["device"] == "cpu"


def test_list_models_prints(mock_weights, capsys):
    rc = run_cli(["list-models"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "unet-r34-4c@v0.0.2" in out


def test_precache_prints_paths(mock_weights, capsys):
    rc = run_cli(["precache", "--weights", "auto:unet-r34-4c@v0.0.2", "--format", "onnx"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Cached weights:" in out
    assert "Cached config:" in out
