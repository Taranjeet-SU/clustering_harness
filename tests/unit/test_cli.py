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
