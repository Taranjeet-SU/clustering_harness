"""Doc steps 12-16: LLM residual classification (variant v1 = C_full) and the refile pass (variant v2 = C_full_v2)."""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fakes import TOPICS, FakeGate, HashEmbedder, fake_llms, make_ctx, make_topic_dataset, rid_of, topic_of  # noqa: E402

from clustering_harness.algorithms.residual import ResidualPass  # noqa: E402
from clustering_harness.config import PipelineConfig  # noqa: E402

NAMES = {"P0": "Password Reset problems", "P1": "Invoice Mismatch problems"}


# =========================================================================================== helpers
def scenario(tmp_path, extra0=0, k2=0, k3=0, n_out=0, residual=None, rules_a=None, rules_b=None, callback_a=None,
             callback_b=None, gate_rule=None, seed=0):
    """Existing clusters P0 (30 x topic 0) and P1 (30 x topic 1); residual = ``extra0`` more topic-0 records,
    ``k2`` topic-2 and ``k3`` topic-3 records and ``n_out`` outliers (dataset order)."""
    ds, X = make_topic_dataset(n_topics=4, per_topic=[30 + extra0, 30, k2, k3], n_outliers=n_out, shuffle=False, seed=seed)
    cfg = PipelineConfig.small()
    cfg.residual = cfg.residual.model_copy(update=residual or {})
    llms, fk = fake_llms(rules_a=rules_a, rules_b=rules_b, callback_a=callback_a, callback_b=callback_b)
    ctx = make_ctx(tmp_path, ds=ds, X=X, cfg=cfg, llms=llms)
    t0 = [t for t, l in zip(ds.ids, ds.labels) if l == TOPICS[0][0]]
    t1 = [t for t, l in zip(ds.ids, ds.labels) if l == TOPICS[1][0]]
    lab = {**{t: "P0" for t in t0[:30]}, **{t: "P1" for t in t1}}
    resid = [t for t in ds.ids if t not in lab]
    gate = FakeGate(gate_rule or pure_rule(ctx))
    return ctx, fk, gate, lab, resid


def pure_rule(ctx):
    def rule(stage, cid, ms):
        ts = {ctx.ds.labels[ctx.pos[t]] for t in ms}
        return ("ACCEPT", 4.5) if len(ts) == 1 else ("REJECT", 2.0)
    return rule


def items_of(info):
    return [rid_of(t) for t in re.findall(r"^\[\d+\] Title: (.*)$", info.user, re.M)]


def cands_of(info):
    return [rid_of(t) for t in info.titles(info.section("Candidate tickets:"))]


def expected_streams(ctx, resid, ns, seed=0, v2=False):
    from sklearn.cluster import KMeans
    order = ctx.sort(resid) if v2 else list(resid)
    random.Random(seed).shuffle(order)
    lab = KMeans(ns, n_init=3, random_state=seed).fit_predict(ctx.V(order)) if ns > 1 else np.zeros(len(order), int)
    return order, lab


# =========================================================================================== streams & batches
def test_streams_shard_shuffled_pool_and_batch_size(tmp_path):
    ctx, fk, gate, lab, resid = scenario(tmp_path, extra0=20, k2=25, k3=25, n_out=10, residual=dict(streams=4))
    ResidualPass(ctx, gate, "v1").run(lab, NAMES, resid, "residual")
    order, slab = expected_streams(ctx, resid, 4)
    files = sorted((ctx.workdir / "cache" / "residual").glob("stream_*.json"))
    assert len(files) == 4 and sum(ctx.load("cache", "residual", f.name)["n"] for f in files) == len(resid)
    cls = fk["a"].of("classify")
    assert all(len(items_of(i)) <= 15 for i in cls)                                 # doc step 12: 15 tickets per call
    s_of = dict(zip(order, slab))
    for s in range(4):
        seq = [t for i in cls if s_of[items_of(i)[0]] == s for t in items_of(i)]
        assert seq == [t for t, k in zip(order, slab) if k == s]                    # one sequential stream per shard
        sizes = [len(items_of(i)) for i in cls if s_of[items_of(i)[0]] == s]
        assert all(n == 15 for n in sizes[:-1])


def test_stream_count_capped_by_pool(tmp_path):
    ctx, fk, gate, lab, resid = scenario(tmp_path, k2=3, residual=dict(streams=4))
    r = ResidualPass(ctx, gate, "v1").run(lab, NAMES, resid, "residual")
    assert len(r["log"]["streams"]) == 3


def test_stream_files_are_reused(tmp_path):
    ctx, fk, gate, lab, resid = scenario(tmp_path, k2=10, k3=10, residual=dict(streams=2))
    ResidualPass(ctx, gate, "v1").run(lab, NAMES, resid, "residual")
    n = len(fk["a"].of("classify"))
    ResidualPass(ctx, gate, "v1").run(lab, NAMES, resid, "residual")
    assert len(fk["a"].of("classify")) == n


def test_classify_prompt_material(tmp_path):
    ctx, fk, gate, lab, resid = scenario(tmp_path, extra0=10, k2=10, residual=dict(streams=1, near_lines=1, desc_chars=50))
    ResidualPass(ctx, gate, "v1").run(lab, NAMES, resid, "residual")
    first = fk["a"].of("classify")[0]
    batch = items_of(first)
    # the nearest existing names to the batch's mean vector (here capped at near_lines=1)
    cent = {c: ctx.centroid([t for t, x in lab.items() if x == c]) for c in NAMES}
    bv = ctx.V(batch).mean(0)
    best = max(cent, key=lambda c: float(cent[c] @ bv))
    assert re.findall(r"^(L\d+)\. (.*)$", first.user, re.M) == [("L1", NAMES[best])]
    assert "(none yet)" in first.user                                               # no provisional labels yet
    descs = re.findall(r"^\s+Description: (.*)$", first.user, re.M)
    assert descs and all(len(d) <= 50 for d in descs)
    assert ctx.prompts["classify_rules_v1"] in first.user


# =========================================================================================== choice parsing
def test_choice_parsing_v1(tmp_path):
    long_label = "export failure for the nightly scheduled job " + "x" * 80

    def classifier(info):
        if info.kind != "classify":
            return None
        b = items_of(info)
        ch = {1: "L1", 2: "NEW: Export failure requests", 3: "NEW: general export questions", 4: "NEW: " + long_label,
              5: "JUNK: service_request", 6: "N1", 8: "L99", 9: "  L1  ", 10: "new: lowercase label"}
        out = [{"n": n, "choice": c} for n, c in ch.items() if n <= len(b)]
        return {"tickets": out + [{"n": 99, "choice": "L1"}, {"n": "x", "choice": "L1"}, {"choice": "L1"}]}
    ctx, fk, gate, lab, resid = scenario(tmp_path, k2=12, residual=dict(streams=1), callback_a=classifier)
    ResidualPass(ctx, gate, "v1").run(lab, NAMES, resid, "residual")
    st = ctx.load("cache", "residual", "stream_0.json")
    info = fk["a"].of("classify")[0]
    b = items_of(info)
    l1 = dict(re.findall(r"^(L\d+)\. (.*)$", info.user, re.M))["L1"]
    l1c = {v: k for k, v in NAMES.items()}[l1]
    assert st["filed"] == {b[0]: l1c, b[8]: l1c}                                    # "L1" and padded "  L1  "
    assert st["prov"] == ["Export failure requests", long_label[:80], "lowercase label"]   # generic label rejected; cut to 80
    assert st["prov_members"] == {"0": [b[1]], "1": [b[3]], "2": [b[9]]}
    assert st["junk"] == {b[4]: "service_request"}
    for t in (b[2], b[5], b[6], b[7]):                                              # generic NEW, N1 not shown, omitted, L99
        assert t not in st["filed"] and t not in st["junk"] and all(t not in v for v in st["prov_members"].values())


@pytest.mark.parametrize("label,rejected", [("General login questions", True), ("misc invoice issues", True),
                                           ("Other export problems", True), ("multiple unrelated asks", True),
                                           ("Mixed bag", True), ("Assorted requests", True), ("various errors", True),
                                           ("Export failure requests", False), ("Generalised ledger export", False)])
def test_generic_label_filter(label, rejected):
    # doc step 12 lists general/misc/other/various/multiple/unrelated/assorted/mixed; the code matches whole words
    from clustering_harness.config import ResidualConfig
    assert bool(re.search(ResidualConfig().generic_label, label, re.I)) == rejected


def test_provisional_growth_uses_creator_title_vector(tmp_path):
    """Doc step 12: the provisional search vector is the title vector of the ticket that created the label,
    not an embedding of the label text. Labels are deliberately written with the *other* topic's words."""
    state = {}

    def classifier(info):
        if info.kind != "classify":
            return None
        b = items_of(info)
        topics = [topic_of(t) for t in re.findall(r"^\[\d+\] Title: (.*)$", info.user, re.M)]
        if "first" not in state:
            state["first"] = b
            i2 = topics.index(TOPICS[2][0])
            i3 = topics.index(TOPICS[3][0])
            state["creators"] = {TOPICS[2][0]: b[i2], TOPICS[3][0]: b[i3]}
            return {"tickets": [{"n": i2 + 1, "choice": f"NEW: {TOPICS[3][0]} tracker"},
                                {"n": i3 + 1, "choice": f"NEW: {TOPICS[2][0]} tracker"}]}
        state["second"] = (b, topics, re.findall(r"^(N\d+)\. (.*)$", info.section("Provisional labels"), re.M))
        return {"tickets": [{"n": 1, "choice": "N1"}]}
    ctx, fk, gate, lab, resid = scenario(tmp_path, k2=8, k3=8, residual=dict(streams=1), callback_a=classifier)
    ResidualPass(ctx, gate, "v1").run(lab, NAMES, resid, "residual")
    b2, topics2, prov_list = state["second"]
    assert len(b2) == 1 and len(prov_list) == 2
    # nearest provisional entry = the label created by the same-topic ticket (whose text names the other topic)
    want = f"{TOPICS[3][0]} tracker" if topics2[0] == TOPICS[2][0] else f"{TOPICS[2][0]} tracker"
    assert prov_list[0] == ("N1", want)
    st = ctx.load("cache", "residual", "stream_0.json")
    k = st["prov"].index(want)
    assert st["prov_members"][str(k)] == [state["creators"][topics2[0]], b2[0]]


# =========================================================================================== canonicalisation
def test_canonicalisation_pregrouping_and_chunks(tmp_path):
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet", "kilo", "lima", "mike"]
    labels_by_ticket = {}

    def classifier(info):
        if info.kind != "classify":
            return None
        out = []
        for n, t in enumerate(items_of(info)):
            lbl = f"{words[int(t) % 13]} {words[(int(t) // 13) % 13]} item"
            labels_by_ticket[t] = lbl
            out.append({"n": n + 1, "choice": f"NEW: {lbl}"})
        return {"tickets": out}

    def canon(info):
        if info.kind != "canonicalise":
            return None
        rows = re.findall(r"^(\d+)\. (.*)$", info.user, re.M)
        # map everything to its first word, but omit odd-numbered rows (fallback = lowercased original label)
        return {"map": [{"n": int(n), "canonical": l.split()[0].upper()} for n, l in rows if int(n) % 2 == 0]}
    ctx, fk, gate, lab, resid = scenario(tmp_path, k2=35, k3=30, residual=dict(streams=1), callback_a=classifier, callback_b=canon,
                                         gate_rule=lambda s, c, m: ("REJECT", 2.0))
    ResidualPass(ctx, gate, "v1").run(lab, NAMES, resid, "residual")
    st = ctx.load("cache", "residual", "stream_0.json")
    labels = st["prov"]
    assert len(labels) == 65
    from sklearn.cluster import AgglomerativeClustering
    LV = HashEmbedder().encode(labels)
    grp = AgglomerativeClustering(None, metric="cosine", linkage="average", distance_threshold=0.2).fit_predict(LV)
    order = np.argsort(grp, kind="stable")
    reqs = fk["b"].of("canonicalise")
    # consecutive chunks of <= 30 labels of the stable group order (chunks run in parallel, so compare as a set)
    shown = sorted(tuple(l for _, l in re.findall(r"^(\d+)\. (.*)$", i.user, re.M)) for i in reqs)
    assert shown == sorted(tuple(labels[k] for k in order[i:i + 30]) for i in range(0, 65, 30))
    assert sorted(len(c) for c in shown) == [5, 30, 30]
    # pooling by exact canonical string: mapped rows -> lowercased first word; omitted rows -> own lowercased label
    canon_of = {}
    for i in reqs:
        for n, l in re.findall(r"^(\d+)\. (.*)$", i.user, re.M):
            canon_of[l] = l.split()[0].lower() if int(n) % 2 == 0 else l.lower()
    pools = {}
    for k, l in enumerate(labels):
        pools.setdefault(canon_of[l], []).extend(st["prov_members"][str(k)])
    gated = gate.items("Cf_gate")
    want = sorted(len(set(v)) for v in pools.values() if len(set(v)) >= 4)
    assert sorted(len(m) for _, m in gated) == want                                          # >= 4 unique members gated
    assert all(re.fullmatch(r"N\d+", c) for c, _ in gated)


def test_min_label_size_and_v2_group_order(tmp_path):
    def classifier(info):
        if info.kind != "classify":
            return None
        out = []
        for n, title in enumerate(re.findall(r"^\[\d+\] Title: (.*)$", info.user, re.M)):
            t = topic_of(title)
            k = int(rid_of(title))
            # topic 2 -> 4 tickets "bbb", 3 tickets "zzz"; topic 3 -> 4 tickets "aaa", rest junk
            lbl = {TOPICS[2][0]: "bbb line" if k % 7 < 4 else "zzz line", TOPICS[3][0]: "aaa line"}.get(t)
            out.append({"n": n + 1, "choice": f"NEW: {lbl}" if lbl else "JUNK: ambiguous"})
        return {"tickets": out}
    ctx, fk, gate, lab, resid = scenario(tmp_path, k2=7, k3=4, residual=dict(refile_streams=1), callback_a=classifier,
                                         callback_b=lambda i: {"map": []} if i.kind == "canonicalise" else None)
    t2 = [t for t in resid if ctx.ds.labels[ctx.pos[t]] == TOPICS[2][0]]
    n_bbb = sum(int(t) % 7 < 4 for t in t2)
    ResidualPass(ctx, gate, "v2").run(lab, NAMES, resid, "refile")
    gated = gate.items("V2_gate")
    sizes = {"bbb line": n_bbb, "zzz line": 7 - n_bbb, "aaa line": 4}
    groups = sorted(sizes.items(), key=lambda x: (-x[1], x[0]))                              # v2: unique count, then name
    want = [(f"RN{i}", n) for i, (name, n) in enumerate(groups) if n >= 4]
    assert [(c, len(m)) for c, m in gated] == want


# =========================================================================================== confirmation, new clusters
def test_confirmation_batches_of_ten_explicit_yes(tmp_path):
    def verifier(info):
        if info.kind == "verify" and info.stage == "Cf_confirm":
            c = cands_of(info)
            return {"answers": [{"n": k + 1, "belongs": int(t) % 3 != 0} for k, t in enumerate(c) if int(t) % 5 != 0]}
    ctx, fk, gate, lab, resid = scenario(tmp_path, extra0=23, residual=dict(streams=1), callback_a=verifier)
    r = ResidualPass(ctx, gate, "v1").run(lab, NAMES, resid, "residual")
    conf = [i for i in fk["a"].of("verify") if i.stage == "Cf_confirm"]
    assert sorted(len(cands_of(i)) for i in conf) == [3, 10, 10]
    p0 = [t for t, c in lab.items() if c == "P0"]
    for i in conf:
        ex = [rid_of(t) for t in i.titles(i.section("Tickets already on this line:", "Candidate tickets:"))]
        assert ex == ctx.exemplars(p0, 3)                                                    # three most central members
        assert i.section("Report line: ", "\n") == NAMES["P0"]
    asked = [t for i in conf for t in cands_of(i)]
    yes = {t for t in asked if int(t) % 3 != 0 and int(t) % 5 != 0}                          # missing answers are not yes
    assert all(r["labels"].get(t) == "P0" and r["how"][t] == "filed_existing_confirmed" for t in yes)
    assert r["log"]["filed_confirmed"] == len(yes)


def test_new_clusters_ids_overlap_and_pruning(tmp_path):
    ctx, fk, gate, lab, resid = scenario(tmp_path, k2=12, k3=12, n_out=4, residual=dict(streams=1))
    r = ResidualPass(ctx, gate, "v1").run(lab, NAMES, resid, "residual")
    new = r["new"]
    assert set(new) == {"CN0", "CN1"}                                                        # ids C + gated id N<i>
    for c, v in new.items():
        assert len({ctx.ds.labels[ctx.pos[t]] for t in v["members"]}) == 1 and len(v["members"]) == 12
        assert all(r["how"][t] == "new_cluster" for t in v["members"])
    outl = [t for t in resid if ctx.ds.labels[ctx.pos[t]] == "outlier"]
    assert all(t not in r["labels"] for t in outl) and set(r["junk"]) == set(outl)
    # pruning sample: seed 11, up to ten members of each new cluster
    checks = [i for i in fk["a"].of("verify") if i.stage == "Cf_p1_check"]
    for i in checks:
        c = cands_of(i)
        cid = next(k for k, v in new.items() if c[0] in v["members"])
        assert c == ctx.sort(random.Random(11).sample(new[cid]["members"], 10))


def test_new_cluster_dissolution_and_double_no(tmp_path):
    def verifier(info):
        if info.kind == "verify" and info.stage == "Cf_p1_check":
            c = cands_of(info)
            t = topic_of(info.titles(info.section("Candidate tickets:"))[0])
            nos = 6 if t == TOPICS[2][0] else 3
            return {"answers": [{"n": k + 1, "belongs": k >= nos} for k in range(len(c))]}

    def rechecker(info):
        if info.kind == "verify" and info.stage == "Cf_p2_recheck":
            c = cands_of(info)
            return {"answers": [{"n": k + 1, "belongs": False} for k in range(len(c) - 1)]}   # last one: missing -> kept
    ctx, fk, gate, lab, resid = scenario(tmp_path, k2=12, k3=12, residual=dict(streams=1), callback_a=verifier, callback_b=rechecker)
    r = ResidualPass(ctx, gate, "v1").run(lab, NAMES, resid, "residual")
    # topic 2: 6 "no", 5 rechecked "no" -> precision 0.5 -> kept minus 5; topic 3: 3 "no", 2 double -> kept minus 2
    assert r["log"]["pruned_dissolved"] == [] and r["log"]["pruned_ejected"] == 7
    sizes = sorted(len(v["members"]) for v in r["new"].values())
    assert sizes == [7, 10]


def test_new_cluster_dissolved_below_half(tmp_path):
    def verifier(info):
        if info.kind == "verify" and info.stage == "Cf_p1_check":
            return {"answers": [{"n": k + 1, "belongs": k >= 6} for k in range(len(cands_of(info)))]}

    def rechecker(info):
        if info.kind == "verify" and info.stage == "Cf_p2_recheck":
            return {"answers": [{"n": k + 1, "belongs": False} for k in range(len(cands_of(info)))]}
    ctx, fk, gate, lab, resid = scenario(tmp_path, k2=12, residual=dict(streams=1), callback_a=verifier, callback_b=rechecker)
    r = ResidualPass(ctx, gate, "v1").run(lab, NAMES, resid, "residual")
    assert r["log"]["pruned_dissolved"] == ["CN0"] and not r["new"]


def test_new_group_drops_confirmed_members(tmp_path):
    """A ticket answered twice in one batch keeps its first answer only (robustness fix): the duplicate filings
    under an existing line are ignored, so nothing is confirmed and the new group keeps all 5 members."""
    def classifier(info):
        if info.kind != "classify":
            return None
        b = items_of(info)
        topics = [topic_of(t) for t in re.findall(r"^\[\d+\] Title: (.*)$", info.user, re.M)]
        lid = dict(re.findall(r"^(L\d+)\. (.*)$", info.user, re.M))
        l0 = next(k for k, v in lid.items() if v == NAMES["P0"])
        out = [{"n": n + 1, "choice": "NEW: export failure requests"} for n, t in enumerate(topics) if t == TOPICS[2][0]]
        # two tickets are also filed under P0 (listed twice; the later filing wins for `filed`)
        dup = [n for n, t in enumerate(topics) if t == TOPICS[2][0]][:2]
        return {"tickets": out + [{"n": n + 1, "choice": l0} for n in dup]}

    def model_a(info):
        if info.kind == "verify":
            return {"answers": [{"n": k + 1, "belongs": True} for k in range(len(cands_of(info)))]}
        return classifier(info)
    ctx, fk, gate, lab, resid = scenario(tmp_path, k2=5, residual=dict(streams=1), callback_a=model_a)
    r = ResidualPass(ctx, gate, "v1").run(lab, NAMES, resid, "residual")
    assert r["log"]["gated"] == 1 and r["log"]["classified_filed_existing"] == 0 and r["log"]["filed_confirmed"] == 0


# regression test (bug fixed after conformance review)
def test_v1_placement_of_unconfirmed_filing_reassigned_later(tmp_path):
    def verifier(info):
        if info.kind == "verify" and info.stage == "Cf_confirm":
            return {"answers": [{"n": k + 1, "belongs": False} for k in range(len(cands_of(info)))]}
    ctx, fk, gate, lab, resid = scenario(tmp_path, extra0=4, residual=dict(streams=1), callback_a=verifier)
    r = ResidualPass(ctx, gate, "v1").run(lab, NAMES, resid, "residual")
    placed = [t for t in resid if r["labels"].get(t) == "P0"]
    assert placed and r["log"]["filed_confirmed"] == 0
    assert all(r["how"][t] == "reassigned" for t in placed)


# =========================================================================================== final reassignment
def test_final_reassignment_excludes_llm_junk_and_labels_placements(tmp_path):
    def classifier(info):
        if info.kind != "classify":
            return None
        out = []
        for n, title in enumerate(re.findall(r"^\[\d+\] Title: (.*)$", info.user, re.M)):
            t = topic_of(title)
            k = int(rid_of(title))
            if t == TOPICS[0][0]:
                out.append({"n": n + 1, "choice": "JUNK: ambiguous" if k % 2 else "JUNK: low_quality_record"})
            # topic 1 extra records: unclassified (left for the centroid reassignment)
        return {"tickets": out}
    ctx, fk, gate, lab, resid = scenario(tmp_path, extra0=6, residual=dict(streams=1), callback_a=classifier,
                                         gate_rule=lambda s, c, m: ("ACCEPT", 4.0))
    # move 6 topic-1 records from P1 into the residual so they are close to P1's centroid
    p1 = [t for t, c in lab.items() if c == "P1"][:6]
    for t in p1:
        lab.pop(t)
    resid = ctx.sort(resid + p1)
    r = ResidualPass(ctx, gate, "v1").run(lab, NAMES, resid, "residual")
    j0 = [t for t in resid if ctx.ds.labels[ctx.pos[t]] == TOPICS[0][0]]
    assert set(r["junk"]) == set(j0) and all(t not in r["labels"] for t in j0)        # explicit LLM-junk never reassigned
    placed = [t for t in p1 if t in r["labels"]]
    assert placed and all(r["labels"][t] == "P1" and r["how"][t] == "reassigned" for t in placed)
    lg = r["log"]["reassign"]
    allm = {}
    for t, c in {**lab}.items():
        allm.setdefault(c, []).append(t)
    own = np.concatenate([ctx.V(v) @ ctx.centroid(v) for v in allm.values()])
    assert lg["tau_hi"] == pytest.approx(np.percentile(own, 25), abs=1e-4) and lg["tau_lo"] == pytest.approx(np.percentile(own, 5), abs=1e-4)


def test_final_reassignment_growth_regate_reverts(tmp_path):
    def classifier(info):
        return {"tickets": []} if info.kind == "classify" else None
    ctx, fk, gate, lab, resid = scenario(tmp_path, residual=dict(streams=1), callback_a=classifier,
                                         gate_rule=lambda s, c, m: ("REJECT", 2.0),
                                         callback_b=lambda i: None)
    p1 = [t for t, c in lab.items() if c == "P1"][:12]
    for t in p1:
        lab.pop(t)
    r = ResidualPass(ctx, gate, "v1").run(lab, NAMES, ctx.sort(resid + p1), "residual")
    rg = gate.items("Cf_regate")
    assert [c for c, _ in rg] == ["P1+cf"]                                            # 18 old members, > 25% growth
    assert r["log"]["reassign"]["reverted"] == 1 and all(t not in r["labels"] for t in p1)


# =========================================================================================== v2 refile pass
def test_v2_prompt_rules_and_junk_reasons(tmp_path):
    ctx, fk, gate, lab, resid = scenario(tmp_path, k2=6, residual=dict(refile_streams=1))
    ResidualPass(ctx, gate, "v2").run(lab, NAMES, resid, "refile")
    P = ctx.prompts
    u = fk["a"].of("classify")[0].user
    assert P["classify_rules_v2"] in u and P["classify_rules_v1"] not in u
    assert "service_request" in P["classify_rules_v1"]
    assert "service_request" not in P["classify_rules_v2"]
    assert re.search(r"reason one of: low_quality_record, insufficient_information, ambiguous", P["classify_rules_v2"])


def test_v2_repair_pass_batches_of_five(tmp_path):
    seen = {"first": set()}

    def classifier(info):
        if info.kind != "classify":
            return None
        b = items_of(info)
        if len(b) > 5 or not seen["first"] >= set(b):                         # main pass: answer only every third ticket
            seen["first"] |= set(b)
            return {"tickets": [{"n": n + 1, "choice": "JUNK: ambiguous"} for n in range(len(b)) if n % 3 == 0]}
        return {"tickets": [{"n": n + 1, "choice": "JUNK: insufficient_information"} for n in range(len(b)) if n != 0]}
    ctx, fk, gate, lab, resid = scenario(tmp_path, k2=30, residual=dict(refile_streams=1), callback_a=classifier)
    r = ResidualPass(ctx, gate, "v2").run(lab, NAMES, resid, "refile")
    reqs = fk["a"].of("classify")
    main = [i for i in reqs if len(items_of(i)) == 15]
    rep = reqs[len(main):]
    missing = [t for i in main for n, t in enumerate(items_of(i)) if n % 3 != 0]
    assert len(main) == 2 and [len(items_of(i)) for i in rep] == [5, 5, 5, 5]
    assert [t for i in rep for t in items_of(i)] == missing                        # stream order, retried once
    cf = r["log"]["classifier_contract_failures"]
    assert cf == {"before_repair": 20, "after_repair": 4}
    assert (ctx.workdir / "cache" / "refile" / "stream_0_repair.json").exists()
    n = len(reqs)
    ResidualPass(ctx, gate, "v2").run(lab, NAMES, resid, "refile")                  # repair file reused
    assert len(fk["a"].of("classify")) == n


def test_v1_has_no_repair_pass(tmp_path):
    ctx, fk, gate, lab, resid = scenario(tmp_path, k2=30, residual=dict(streams=1),
                                         callback_a=lambda i: {"tickets": []} if i.kind == "classify" else None)
    r = ResidualPass(ctx, gate, "v1").run(lab, NAMES, resid, "residual")
    assert len(fk["a"].of("classify")) == 2 and "classifier_contract_failures" not in r["log"]
    assert not list((ctx.workdir / "cache" / "residual").glob("*repair*"))


def test_v2_ids_placements_and_order_independence(tmp_path):
    out = []
    for k, rev in enumerate((False, True)):
        ctx, fk, gate, lab, resid = scenario(tmp_path / str(k), extra0=8, k2=12, k3=12, n_out=4, residual=dict(refile_streams=2))
        r = ResidualPass(ctx, gate, "v2").run(lab, NAMES, list(reversed(resid)) if rev else resid, "refile")
        out.append(r)
    a, b = out
    assert a["labels"] == b["labels"] and a["how"] == b["how"] and a["junk"] == b["junk"]   # ids sorted before the shuffle
    assert set(a["new"]) == {"CRN0", "CRN1"}
    hows = set(a["how"].values())
    assert hows <= {"refiled_filed_existing_confirmed", "refiled_new_cluster", "refiled_reassigned"}
    assert "refiled_new_cluster" in hows and "refiled_filed_existing_confirmed" in hows
    # placements only for newly placed records; old members keep no placement here (the pipeline preserves theirs)
    assert all(t not in a["how"] for t in a["labels"] if t in lab)
    assert {t for t in a["labels"] if t not in lab} == set(a["how"])


def test_v1_order_follows_input(tmp_path):
    ctx, fk, gate, lab, resid = scenario(tmp_path, k2=10, residual=dict(streams=1))
    ResidualPass(ctx, gate, "v1").run(lab, NAMES, list(reversed(resid)), "residual")
    order = list(reversed(resid))
    random.Random(0).shuffle(order)
    assert items_of(fk["a"].of("classify")[0]) == order[:10]


def test_reference_residual_settings():
    r = PipelineConfig.reference().residual
    assert (r.streams, r.batch_size, r.repair_batch_size, r.near_lines, r.near_provisional, r.desc_chars) == (4, 15, 5, 30, 30, 220)
    assert (r.canon_distance, r.canon_chunk, r.min_label_size, r.confirm_batch, r.seed, r.refile_streams) == (0.2, 30, 4, 10, 0, 24)
