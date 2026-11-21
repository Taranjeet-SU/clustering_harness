"""Prompt templates.

All prompts sent by the pipeline are rendered from a :class:`PromptSet`: a flat mapping of template names to
strings with ``{{placeholder}}`` slots (double braces, so JSON examples inside prompts need no escaping).
Prompt sets ship as YAML presets in ``clustering_harness/presets``:

* ``default`` — domain-neutral wording; fill ``{{product}}`` / ``{{product_description}}`` via
  :meth:`PromptSet.load` (``domain=...``) or override any template.
* ``searchunify`` — the prompts of the reference SearchUnify run (an example of a fully adapted domain).

To adapt the pipeline to a new domain, copy ``default.yaml``, rewrite the domain sentences and the gate's
anchor examples (they calibrate what "one report line" means), and load it with ``PromptSet.from_yaml``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import yaml

__all__ = ["PromptSet", "REQUIRED_TEMPLATES", "list_presets"]

REQUIRED_TEMPLATES = (
    "item", "item_empty_desc",
    "gate_system", "gate_user",
    "verify_system", "verify_user", "verify_default_criterion",
    "label_system", "label_user",
    "merge_system", "merge_user",
    "classify_system", "classify_user", "classify_rules_v1", "classify_rules_v2", "classify_item", "classify_no_provisional",
    "canon_user",
    "judge_system", "judge_rubric", "judge_user", "judge_item", "judge_empty_desc",
    "score_system_suffix", "naming_task", "judgment_task", "label_fit_rubric", "distinction_rubric",
    "rollup_system", "rollup_rubric", "rollup_decide_user", "rollup_row", "rollup_name_user",
)

_SLOT = re.compile(r"\{\{(\w+)\}\}")


def list_presets() -> list[str]:
    return sorted(p.name[:-5] for p in resources.files("clustering_harness.presets").iterdir() if p.name.endswith(".yaml"))


@dataclass
class PromptSet:
    templates: dict = field(default_factory=dict)
    name: str = "custom"
    #: the domain values and the templates before they were filled in (what :meth:`to_yaml` writes)
    domain: dict = field(default_factory=dict)
    raw_templates: dict = field(default_factory=dict)

    def __post_init__(self):
        missing = [k for k in REQUIRED_TEMPLATES if k not in self.templates]
        if missing:
            raise ValueError(f"prompt set {self.name!r} is missing templates: {missing}")

    def render(self, key: str, **values) -> str:
        """Fill ``{{slot}}`` placeholders; every slot used by the template must be supplied."""
        tpl = self.templates[key]

        def sub(m):
            k = m.group(1)
            if k not in values:
                raise KeyError(f"template {key!r} needs value {k!r}")
            return str(values[k])
        return _SLOT.sub(sub, tpl)

    def __getitem__(self, key: str) -> str:
        return self.templates[key]

    def fingerprint(self) -> str:
        """Hash of the rendered templates (prompt file edits and domain values both change it)."""
        import hashlib
        import json
        return hashlib.sha256(json.dumps(self.templates, sort_keys=True).encode()).hexdigest()[:16]

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_yaml(cls, path: str | Path, domain: dict | None = None, overrides: dict | None = None) -> "PromptSet":
        data = yaml.safe_load(Path(path).read_text())
        return cls._build(data, Path(path).stem, domain, overrides)

    @classmethod
    def load(cls, preset: str = "default", domain: dict | None = None, overrides: dict | None = None) -> "PromptSet":
        """Load a built-in preset by name, or a YAML file by path."""
        if preset.endswith((".yaml", ".yml")) or "/" in preset:
            return cls.from_yaml(preset, domain, overrides)
        try:
            text = resources.files("clustering_harness.presets").joinpath(f"{preset}.yaml").read_text()
        except (FileNotFoundError, OSError):
            import difflib
            near = difflib.get_close_matches(preset, list_presets(), 1)
            raise ValueError(f"unknown prompt preset {preset!r}" + (f" (did you mean {near[0]!r}?)" if near else "")
                             + f"; built-in presets: {', '.join(list_presets())}, or give a path to a prompt YAML") from None
        return cls._build(yaml.safe_load(text), preset, domain, overrides)

    @classmethod
    def _build(cls, data: dict, name: str, domain: dict | None, overrides: dict | None) -> "PromptSet":
        tpls = dict(data.get("templates", data))
        dom = dict(data.get("domain", {}))
        dom.update(domain or {})
        tpls.update(overrides or {})
        used = set(dict(data.get("domain", {}))) | {m for v in tpls.values() for m in _SLOT.findall(str(v))}
        unknown = [k for k in (domain or {}) if k not in used]
        if unknown:
            import difflib
            hint = "; ".join(f"{k!r}" + (f" (did you mean {m[0]!r}?)" if (m := difflib.get_close_matches(k, sorted(used), 1)) else "")
                             for k in unknown)
            raise ValueError(f"domain key(s) not used by prompt set {name!r}: {hint}. Keys it uses: "
                             f"{', '.join(sorted(dict(data.get('domain', {}))))}")
        raw = dict(tpls)
        # domain variables are substituted once at load time; data slots stay for render()
        if dom:
            pat = re.compile(r"\{\{(" + "|".join(map(re.escape, dom)) + r")\}\}")
            tpls = {k: pat.sub(lambda m: str(dom[m.group(1)]), v) for k, v in tpls.items()}
        return cls(templates=tpls, name=name, domain=dom, raw_templates=raw)

    def to_yaml(self, path: str | Path) -> None:
        """Write an editable prompt file: the ``domain`` block and the templates with their ``{{...}}`` slots, so it
        loads back with ``PromptSet.load(path, domain=...)`` and a spec's ``domain:`` still applies."""
        data = {"domain": dict(self.domain), "templates": dict(self.raw_templates or self.templates)}
        Path(path).write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=10_000))
