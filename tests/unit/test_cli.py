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
    with pytest.raises(FileNotFoundError):            # the data file is missing; the spec folder is on sys.path by now
        main(["run", str(spec)])
    from clustering_harness.factory import build_backend
    assert build_backend({"backend": "my_plugin_mod:Echo"}).model_id == "echo:e"
