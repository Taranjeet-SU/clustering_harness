"""Command-line entry points that need no models or network."""
import pytest

from clustering_harness.cli import main


@pytest.mark.parametrize("cmd", [[], ["init"], ["run"], ["compare"], ["multirun"], ["evaluate"], ["datasets"]])
def test_help_works_for_every_command(cmd, capsys):
    with pytest.raises(SystemExit) as e:
        main(cmd + ["--help"])
    assert e.value.code == 0
    assert "usage" in capsys.readouterr().out


def test_datasets_lists_repositories(capsys):
    assert main(["datasets"]) == 0
    out = capsys.readouterr().out
    assert "PolyAI/banking77" in out


def test_init_prints_a_loadable_spec(capsys):
    import yaml
    assert main(["init", "--preset", "small"]) == 0
    spec = yaml.safe_load(capsys.readouterr().out)
    assert {"data", "pipeline", "models", "embedder"} <= set(spec)


def test_until_typo_is_rejected(tmp_path, capsys):
    spec = tmp_path / "s.yaml"
    spec.write_text("workdir: w\ndata: {path: none.csv}\nmodels: {}\n")
    with pytest.raises(SystemExit) as e:
        main(["run", str(spec), "--until", "bnak"])
    assert e.value.code == 2 and "invalid choice" in capsys.readouterr().err


def test_plugins_next_to_the_spec_are_importable(tmp_path, monkeypatch):
    import sys
    (tmp_path / "my_plugin_mod.py").write_text(
        "from clustering_harness.llm import ChatResult\n"
        "class Echo:\n"
        "    def __init__(self, model='e'):\n"
        "        self.model, self.model_id = model, 'echo:' + model\n"
        "    def complete(self, request):\n"
        "        return ChatResult(text='{}')\n")
    spec = tmp_path / "s.yaml"
    spec.write_text("workdir: w\ndata: {path: missing.csv}\nmodels: {a: {backend: 'my_plugin_mod:Echo'}, b: {backend: 'my_plugin_mod:Echo'}}\n")
    monkeypatch.chdir("/")
    monkeypatch.setattr(sys, "path", list(sys.path))
    assert main(["run", str(spec)]) == 2               # the data file is missing; the spec folder is on sys.path by now
    from clustering_harness.factory import build_backend
    assert build_backend({"backend": "my_plugin_mod:Echo"}).model_id == "echo:e"


def _spec(tmp_path, body):
    p = tmp_path / "s.yaml"
    p.write_text(body)
    return str(p)


MODELS = "models: {a: {backend: offline, model_id: x}, b: {backend: offline, model_id: y}}\n"


@pytest.mark.parametrize("body,needle", [
    ("workdir: w\ndata: {path: d.csv}\n" + MODELS + "embeder: {backend: hf}\n", "'embeder' (did you mean 'embedder'?)"),
    ("workdir: w\ndata: {path: d.csv, text_column: t}\n" + MODELS, "'text_column' (did you mean 'text_col'?)"),
    ("workdir: w\ndata: {path: d.csv}\npipeline: {preset: smal}\n" + MODELS, "pipeline.preset 'smal'"),
    ("workdir: w\ndata: {path: d.csv}\nmodels: {a: {backend: offline, model_id: x}}\n", "missing ['b']"),
    ("workdir: w\ndata: {path: d.csv, dataset: banking77}\n" + MODELS, "exactly one of"),
    ("workdir: w\ndata: {mongo: {collection: t, text_feild: x}}\n" + MODELS, "'text_feild' (did you mean 'text_field'?)"),
])
def test_spec_typos_are_one_line_errors(tmp_path, capsys, body, needle):
    assert main(["run", _spec(tmp_path, body)]) == 2
    err = capsys.readouterr().err
    assert needle in err and "Traceback" not in err


def test_unknown_baseline_fails_before_the_pipeline(tmp_path, capsys):
    (tmp_path / "d.csv").write_text("id,text\n1,a\n")
    s = _spec(tmp_path, f"workdir: {tmp_path / 'w'}\ndata: {{path: d.csv}}\n" + MODELS)
    assert main(["compare", s, "--baselines", "leiden"]) == 2
    assert "unknown baseline 'leiden'" in capsys.readouterr().err
    assert not (tmp_path / "w").exists()                           # nothing ran
    assert main(["compare", s, "--baselines", "kmeans:kk=7"]) == 2
    assert "'kk'" in capsys.readouterr().err


def test_data_path_is_found_next_to_the_spec(tmp_path, monkeypatch):
    from clustering_harness.cli import load_spec
    (tmp_path / "d.csv").write_text("id,text\n1,a\n")
    s = _spec(tmp_path, "data: {path: d.csv}\n" + MODELS)
    monkeypatch.chdir("/")
    assert load_spec(s)["data"]["path"] == str(tmp_path / "d.csv")
