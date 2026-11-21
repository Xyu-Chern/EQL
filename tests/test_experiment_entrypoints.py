import json
from pathlib import Path

from eql.records import verify_source
from experiments import cgpo


def test_archived_baseline_sources_verify():
    root = Path(__file__).resolve().parents[1]
    manifest = root / "results/original_comparison/Pendulum-v1/EQL_manifest.json"
    for filename, expected in json.loads(manifest.read_text())["source_sha256"].items():
        verify_source(filename, expected)


def test_cgpo_uses_archived_runner_with_requested_output(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(cgpo.subprocess, "run", lambda args, **options: calls.append((args, options)))
    cgpo.run("Walker2d-v3", 204, tmp_path)
    (args, options), = calls
    assert Path(args[2]).is_file()
    assert args[3:] == ["--env", "Walker2d-v3", "--seed", "204", "--cuda", "cuda:0", "--aug", "--weighted", "--running_v"]
    assert options["env"]["CGPO_RESULTS"] == str(tmp_path.resolve())
