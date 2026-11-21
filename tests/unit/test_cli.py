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


def test_domain_and_prompt_typos(tmp_path, capsys):
    base = "workdir: w\ndata: {path: d.csv}\n" + MODELS
    assert main(["run", _spec(tmp_path, base + "pipeline: {domain: {prodct: X}}\n")]) == 2
    assert "'prodct' (did you mean 'product'?)" in capsys.readouterr().err
    assert main(["run", _spec(tmp_path, base + "pipeline: {prompts: defualt}\n")]) == 2
    assert "unknown prompt preset 'defualt' (did you mean 'default'?)" in capsys.readouterr().err
    assert main(["run", _spec(tmp_path, base + "pipeline: {preset: refrence}\n")]) == 2
    assert "(did you mean 'reference'?)" in capsys.readouterr().err


def test_reset_command_keeps_model_answers(tmp_path, capsys):
    wd = tmp_path / "w"
    for f in ("cache/llm.sqlite", "spend.sqlite", "manifest.json", "results/base/labels.npy", "output/summary.json",
              "results/rollup/lines.csv", "cache/labels/l.npy"):
        (wd / f).parent.mkdir(parents=True, exist_ok=True)
        (wd / f).write_text("x")
    s = _spec(tmp_path, f"workdir: {wd}\ndata: {{path: d.csv}}\n" + MODELS)
    assert main(["reset", s, "--from-stage", "rollup"]) == 0
    assert not (wd / "results/rollup").exists() and not (wd / "output").exists() and (wd / "results/base/labels.npy").exists()
    assert main(["reset", s]) == 0
    assert sorted(str(p.relative_to(wd)) for p in wd.rglob("*") if p.is_file()) == ["cache/labels/l.npy", "cache/llm.sqlite",
                                                                                   "spend.sqlite"]


def test_misspelled_model_option_gets_a_suggestion(tmp_path, capsys):
    (tmp_path / "d.csv").write_text("id,text\n1,a\n")
    s = _spec(tmp_path, "workdir: w\ndata: {path: d.csv}\nmodels: {a: {backend: openai, model: gpt-4o-mini, budget_usdd: 1}, "
                        "b: {backend: offline, model_id: y}}\n")
    assert main(["run", s]) == 2
    assert "'budget_usdd' (did you mean 'budget_usd'?)" in capsys.readouterr().err


def test_evaluate_fails_when_no_ids_match(tmp_path, capsys):
    (tmp_path / "a.csv").write_text("ticket_id,cluster_id\nx1,c1\nx2,c2\n")
    (tmp_path / "g.csv").write_text("id,label\n1,a\n2,b\n")
    assert main(["evaluate", "--assignments", str(tmp_path / "a.csv"), "--gold", str(tmp_path / "g.csv")]) == 2
    assert "no record ids match" in capsys.readouterr().err


def test_pipeline_typos_get_suggestions(tmp_path, capsys):
    (tmp_path / "d.csv").write_text("id,text\n1,a\n")
    s = _spec(tmp_path, "workdir: w\ndata: {path: d.csv}\npipeline: {gate: {accept_men: 3}}\n" + MODELS)
    assert main(["run", s]) == 2
    assert "'pipeline.gate.accept_men' (did you mean 'pipeline.gate.accept_mean'?)" in capsys.readouterr().err


def test_wrong_csv_column_is_named(tmp_path, capsys):
    (tmp_path / "d.csv").write_text("id,body\n1,a\n")
    s = _spec(tmp_path, "workdir: w\ndata: {path: d.csv, text_col: bdy}\n" + MODELS)
    assert main(["run", s]) == 2
    assert "column 'bdy' not found (did you mean 'body'?)" in capsys.readouterr().err


def test_prompt_file_round_trip_keeps_the_domain(tmp_path):
    from clustering_harness.prompts import PromptSet
    f = tmp_path / "p.yaml"
    PromptSet.load("default").to_yaml(f)
    assert PromptSet.load(str(f), domain={"product": "Acme"}).fingerprint() == \
        PromptSet.load("default", domain={"product": "Acme"}).fingerprint()
