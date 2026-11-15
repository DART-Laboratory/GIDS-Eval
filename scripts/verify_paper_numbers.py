from __future__ import annotations

import argparse
import csv
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

DS = {"0501": "Campus", "lanl": "LANL", "cic": "CI-2017", "optc": "OpTC"}
DIMS = ["featurization", "graph_encoder", "temporal_encoder", "loss"]

results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    results.append((ok, name, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail and not ok else ""))
    return ok


def close(a: float, b: float, tol: float = 5e-4) -> bool:
    return abs(a - b) <= tol


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def newest(base: Path, prefix: str = "") -> Path:
    cands = sorted(d for d in base.iterdir() if d.is_dir() and d.name.startswith(prefix))
    if not cands:
        raise SystemExit(f"no result directory under {base}")
    return cands[-1]


def verify_eg1(code: Path) -> None:
    print("\nEG1  snapshot-window sensitivity")
    d = newest(code / "EG1/results", "2026-04-01")
    pairs = read(d / "evidence_table_pairs.csv")

    swings = [float(r["ap_swing_rel"]) for r in pairs if r.get("ap_swing_rel") not in ("", None)]
    mean_swing = 100 * st.mean(swings)
    check(close(mean_swing, 38.3, 0.05), "mean relative AP swing = 38.3%", f"got {mean_swing:.2f}%")
    over20 = sum(1 for s in swings if s > 0.20)
    check(over20 == 11 and len(swings) == 16,
          "11 of 16 pairs exceed a 20% relative AP swing", f"got {over20} of {len(swings)}")

    raw = read(d / "raw_results.csv")
    ao = {r["window_label"]: float(r["ap"]) for r in raw
          if r["detector"] == "argus" and r["dataset"] == "optc"}
    check(close(ao.get("6m", -1), 0.760, 5e-4) and close(ao.get("150m", -1), 0.033, 5e-4),
          "Argus/OpTC falls 0.760 (6m) -> 0.033 (150m)", f"got {ao}")
    check(close(ao.get("1h", -1), 0.334, 5e-4),
          "Argus/OpTC 1h = 0.334, the EG8 native baseline", f"got {ao.get('1h')}")

    pk = [r for r in pairs if r["detector"] == "pikachu" and r["dataset"] == "cic"]
    check(bool(pk) and pk[0].get("default_mean_ap", "") == "",
          "Pikachu/CI-2017 has no default-window AP (table prints a dash)")


def verify_eg5(code: Path) -> None:
    print("\nEG5  cost and scaling")
    d = newest(code / "EG5/results")
    raw = read(d / "raw_results.csv")
    by = defaultdict(dict)
    for r in raw:
        by[(r["detector"], r["dataset"])][int(r["scale_percent"])] = r

    growth = {k: float(v[100]["total_wall_time_sec"]) / float(v[25]["total_wall_time_sec"])
              for k, v in by.items() if 25 in v and 100 in v
              and float(v[25]["total_wall_time_sec"]) > 0}
    worst, g = max(growth.items(), key=lambda kv: kv[1])
    check(worst == ("pikachu", "lanl") and close(g, 14.9, 0.05),
          "largest 25->100% runtime growth = 14.9x on Pikachu/LANL",
          f"got {worst} {g:.2f}x")

    pk = by[("pikachu", "0501")]
    gb = [float(pk[s]["run_peak_rss_gib"]) * 1.073741824 for s in (25, 50, 75, 100)]
    check(all(close(a, b, 0.05) for a, b in zip(gb, [2.8, 3.4, 3.0, 3.5])),
          "Pikachu/Campus memory row is in decimal GB",
          f"got {[round(x, 1) for x in gb]}")

    check(all(float(r["run_peak_rss_gib"]) > 0 for r in raw),
          "every EG5 row, including Anomal-E, reports measured peak RSS")

    table14 = {"0501": ([2517.847875, 5381.059735, 8156.34276, 11705.365625],
                        [10.6, 20.6, 30.5, 40.6], 1.1),
               "cic": ([1752.151596, 3293.171859, 5318.923534, 6676.558837],
                       [10.8, 20.9, 30.9, 41.0], 1.0)}
    for ds, (secs, gbs, exp) in table14.items():
        an = by[("anomal", ds)]
        t = [float(an[s]["total_wall_time_sec"]) for s in (25, 50, 75, 100)]
        m = [float(an[s]["run_peak_rss_gib"]) * 1.073741824 for s in (25, 50, 75, 100)]
        slope = math.log(t[3] / t[0]) / math.log(4)
        check(all(close(a, b, 0.01) for a, b in zip(t, secs)),
              f"Anomal-E/{DS[ds]} wall-clock row matches the rerun",
              f"got {[round(x, 1) for x in t]}")
        check(all(close(a, b, 0.05) for a, b in zip(m, gbs)),
              f"Anomal-E/{DS[ds]} memory row is in decimal GB",
              f"got {[round(x, 1) for x in m]}")
        check(close(slope, exp, 0.05),
              f"Anomal-E/{DS[ds]} scaling exponent = {exp}", f"got {slope:.2f}")

    mem_max = max(raw, key=lambda r: float(r["run_peak_rss_gib"]))
    check((mem_max["detector"], mem_max["dataset"]) == ("pikachu", "cic")
          and close(float(mem_max["run_peak_rss_gib"]) * 1.073741824, 67.9, 0.1),
          "scale-experiment peak memory = 67.9 GB on Pikachu/CI-2017",
          f"got {mem_max['detector']}/{mem_max['dataset']} "
          f"{float(mem_max['run_peak_rss_gib']) * 1.073741824:.1f}")

    mem_growth = {k: float(v[100]["run_peak_rss_gib"]) / float(v[25]["run_peak_rss_gib"])
                  for k, v in by.items() if 25 in v and 100 in v
                  and float(v[25]["run_peak_rss_gib"]) > 0}
    worst_m, gm = max(mem_growth.items(), key=lambda kv: kv[1])
    check(worst_m == ("pikachu", "cic") and close(gm, 4.5, 0.05),
          "largest 25->100% memory growth = 4.5x on Pikachu/CI-2017",
          f"got {worst_m} {gm:.2f}x")


def verify_eg6(code: Path) -> None:
    print("\nEG6  covering-edge evasion")
    d = newest(code / "EG6/results", "2026-08-04")
    ev = {(r["detector"], r["dataset"]): r for r in read(d / "evidence_table.csv")}

    expect = {
        ("argus", "0501"): (0, None, ""), ("argus", "lanl"): (352, 1.0, "2"),
        ("argus", "cic"): (145, 0.0, ""), ("argus", "optc"): (1053, 0.1415, "20"),
        ("euler", "0501"): (0, None, ""), ("euler", "lanl"): (407, 0.9484, "5"),
        ("euler", "cic"): (0, None, ""), ("euler", "optc"): (1059, 0.3173, "10"),
        ("vgrnn", "0501"): (0, None, ""), ("vgrnn", "lanl"): (371, 1.0, "2"),
        ("vgrnn", "cic"): (3, 1.0, "2"), ("vgrnn", "optc"): (96, 0.125, ""),
    }
    ok = True
    for k, (flag, asr2, full) in expect.items():
        r = ev.get(k)
        if r is None:
            ok = False
            continue
        got_flag = int(float(r["flagged_target_count"]))
        got_full = r["full_evasion_budget"]
        got_asr = r["attack_success_rate_k2"]
        good = got_flag == flag and got_full == full and (
            got_asr == "" if asr2 is None else close(float(got_asr), asr2, 1e-3))
        if not good:
            ok = False
            print(f"      {k}: flag={got_flag}/{flag} full={got_full!r}/{full!r} asr@2={got_asr}")
    check(ok, "all 12 pairs: flagged count, ASR@2 and full-evasion budget")

    zero = [k for k, r in ev.items() if int(float(r["flagged_target_count"])) == 0]
    meas = [k for k, r in ev.items() if int(float(r["flagged_target_count"])) > 0]
    within20 = [k for k in meas if ev[k]["full_evasion_budget"] not in ("",)
                and int(ev[k]["full_evasion_budget"]) <= 20]
    at2 = [k for k in meas if ev[k]["full_evasion_budget"] == "2"]
    check(len(zero) == 4, "4 of 12 pairs flag no attack edge", f"got {len(zero)}")
    check(len(meas) == 8, "8 measurable pairs", f"got {len(meas)}")
    check(len(within20) == 6, "6 of 8 reach full evasion within 20 edges", f"got {len(within20)}")
    check(len(at2) == 3, "3 reach full evasion at K=2", f"got {len(at2)}")

    raw = read(d / "raw_results.csv")
    bad = [r for r in raw if r["attack_success_rate"] not in ("",)
           and int(float(r["budget"])) == 0 and float(r["attack_success_rate"]) > 0]
    check(not bad, "ASR at budget 0 is zero (pre-existing misses are excluded)",
          f"{len(bad)} rows non-zero")


def _eg7_means(raw: list[dict]) -> dict:
    means = {}
    for ds in DS:
        sub = [r for r in raw if r["dataset"] == ds]
        for dim in DIMS:
            g = defaultdict(list)
            for r in sub:
                try:
                    g[r[dim]].append(float(r["ap"]))
                except ValueError:
                    pass
            means[(ds, dim)] = sorted(((k, st.mean(v)) for k, v in g.items()), key=lambda x: -x[1])
    return means


def verify_eg7(code: Path) -> None:
    print("\nEG7  component attribution")
    d = newest(code / "EG7/results", "2026-03-31")
    raw = read(d / "raw_results.csv")
    check(len(raw) == 960, "960 grid runs", f"got {len(raw)}")
    cfgs = {(r["featurization"], r["graph_encoder"], r["temporal_encoder"], r["loss"])
            for r in raw if r["dataset"] == "cic"}
    check(len(cfgs) == 240, "240 configurations per dataset (\\numconfigs)", f"got {len(cfgs)}")

    m = _eg7_means(raw)
    best = {(ds, dim): m[(ds, dim)][0] for ds in DS for dim in DIMS}
    check(best[("lanl", "graph_encoder")][0] == "none"
          and close(best[("lanl", "graph_encoder")][1], 0.254, 1e-3),
          "LANL best graph encoder is none (.254)")
    optc = dict(m[("optc", "graph_encoder")])
    check(max(optc.values()) - optc["none"] < 0.005,
          "OpTC: encoder-free within 0.005 AP of the best option")
    check(all(best[(ds, "loss")][0] == "reconstruction" for ds in ("0501", "cic", "lanl"))
          and best[("optc", "loss")][0] == "link_prediction",
          "reconstruction wins on Campus/CI-2017/LANL, link prediction on OpTC")

    camp = dict(m[("0501", "graph_encoder")])
    check(best[("0501", "graph_encoder")][0] == "mpnn"
          and close(best[("0501", "graph_encoder")][1], 0.269, 1e-3),
          "Campus best graph encoder is MPNN (.269)",
          f"got {best[('0501', 'graph_encoder')]}")
    check(close(camp["none"], 0.127, 1e-3) and camp["none"] == min(camp.values()),
          "Campus encoder-free mean is .127, last of five",
          f"got {camp['none']:.3f}")
    check(best[("0501", "featurization")][0] == "one_hot_structural"
          and close(best[("0501", "featurization")][1], 0.216, 1e-3),
          "Campus best featurization is identity+structural (.216)",
          f"got {best[('0501', 'featurization')]}")
    check(best[("0501", "temporal_encoder")][0] == "gru"
          and close(best[("0501", "temporal_encoder")][1], 0.198, 1e-3),
          "Campus best temporal encoder is GRU (.198)",
          f"got {best[('0501', 'temporal_encoder')]}")
    check(close(dict(m[("0501", "loss")])["reconstruction"], 0.427, 1e-3),
          "Campus reconstruction mean is .427")

    lite = {r["dataset"]: r for r in raw
            if r["featurization"] == "one_hot_structural" and r["graph_encoder"] == "none"
            and r["temporal_encoder"] == "none" and r["loss"] == "reconstruction"}
    check(len(lite) == 4, "the GIDS-Lite configuration is present for all four datasets")
    check(close(float(lite["0501"]["ap"]), 0.322, 1e-3)
          and close(float(lite["lanl"]["ap"]), 0.510, 1e-3),
          "Table 4 Campus (0.322) and LANL (0.510) reproduce from the grid")


def verify_detection_cic(code: Path) -> None:
    print("\nTable 4  GIDS-Lite CI-2017 operating point")
    d = code / "EG7/results/2026-08-05-gidslite-cic-rerun"
    if not (d / "baseline_metrics.csv").exists():
        check(False, "CI-2017 rerun result set present", f"missing {d}")
        return
    r = read(d / "baseline_metrics.csv")[0]
    cfg = (d / "job_config.yaml").read_text(encoding="utf-8")
    check("window_seconds: 3600" in cfg and "one_hot_structural" in cfg
          and "loss: reconstruction" in cfg,
          "rerun used the reported configuration at a one-hour window")
    for name, col, want in (("AP", "ap", 0.970), ("CDR", "cdr", 0.714),
                            ("precision", "precision", 0.974), ("FPR", "fpr", 0.011)):
        check(close(float(r[col]), want, 5e-4),
              f"Table 4 CI-2017 {name} = {want}", f"got {float(r[col]):.6f}")
    tp, fp, fn, tn = (int(r[k]) for k in ("tp", "fp", "fn", "tn"))
    check(tp + fn == 332928 and fp + tn == 762531,
          "test split matches Appendix C (762,531 benign + 332,928 attack)",
          f"got {fp + tn} + {tp + fn}")

    grid = newest(code / "EG7/results", "2026-03-31")
    g = [x for x in read(grid / "raw_results.csv")
         if x["dataset"] == "cic" and x["featurization"] == "one_hot_structural"
         and x["graph_encoder"] == "none" and x["temporal_encoder"] == "none"
         and x["loss"] == "reconstruction"]
    check(bool(g) and not close(float(g[0]["ap"]), float(r["ap"]), 1e-3),
          "the EG7 grid row is a different window, so it differs by design",
          f"grid {float(g[0]['ap']):.4f} vs operating point {float(r['ap']):.4f}")


def verify_window(code: Path) -> None:
    print("\nEG1/GIDS-Lite  window sweep")
    d = code / "EG7/results/2026-08-05-gidslite-window-singleconfig"
    raw = read(d / "raw_results.csv")
    by = defaultdict(list)
    for r in raw:
        by[r["dataset"]].append(r)

    lite = ("one_hot_structural", "reconstruction")
    check(all((r["featurization"], r["loss"]) == lite for r in raw),
          "every sweep row uses the reported configuration")

    expect = {"0501": (0.278, 0.354, 21.3, 0.0, "450s"),
              "lanl": (0.112, 0.510, 78.1, 23.5, "1h"),
              "cic": (0.980, 0.988, 0.8, 0.0, "10m"),
              "optc": (0.976, 0.993, 1.8, 0.0, "150m")}
    for ds, (lo, hi, sw, dc, bw) in expect.items():
        aps = [(float(r["ap"]), r["window_label"]) for r in by[ds]]
        cdrs = [float(r["cdr"]) for r in by[ds]]
        mn, mx = min(aps), max(aps)
        swing = 100 * (mx[0] - mn[0]) / mx[0]
        dcdr = 100 * (max(cdrs) - min(cdrs))
        check(close(mn[0], lo, 5e-4) and close(mx[0], hi, 5e-4)
              and close(swing, sw, 0.05) and close(dcdr, dc, 0.05) and mx[1] == bw,
              f"{DS[ds]} row (range, swing, dCDR, best window)",
              f"got {mn[0]:.3f}--{mx[0]:.3f}, {swing:.1f}%, {dcdr:.1f}pp, best {mx[1]}")

    camp = {r["window_label"]: r for r in by["0501"]}
    lanl = {r["window_label"]: r for r in by["lanl"]}
    check(close(float(camp["5m"]["ap"]), 0.322218, 1e-6)
          and close(float(lanl["1h"]["ap"]), 0.509901, 1e-6),
          "the 5m Campus and 1h LANL rows reproduce the EG7 grid cells exactly")


def verify_eg8(code: Path) -> None:
    print("\nEG8  native vs aligned preprocessing")
    d = newest(code / "EG8/results")
    raw = {(r["protocol"], r["detector"], r["dataset"]): r for r in read(d / "raw_results.csv")}

    def ap(proto, det, ds):
        return float(raw[(proto, det, ds)]["ap"])

    check(close(ap("native", "euler", "lanl"), 0.035, 1e-3)
          and close(ap("aligned", "euler", "lanl"), 0.423, 1e-3),
          "Euler/LANL moves .035 -> .423 under the aligned contract")
    delta = ap("aligned", "vgrnn", "lanl") - ap("native", "vgrnn", "lanl")
    check(close(100 * delta, 61.8, 0.05), "largest AP shift is 61.8 pp (VGRNN/LANL)",
          f"got {100 * delta:.2f}")
    d_pk = 100 * (ap("aligned", "pikachu", "0501") - ap("native", "pikachu", "0501"))
    check(close(d_pk, 1.9, 0.05), "Pikachu/Campus delta is 1.9 pp", f"got {d_pk:.2f}")
    check(close(ap("native", "argus", "optc"), 0.334, 1e-3),
          "Argus/OpTC native (1h window) = 0.334")

    wins = defaultdict(set)
    for (proto, det, ds), r in raw.items():
        if r["window_seconds"]:
            wins[ds].add(r["window_seconds"])
    check(all(len(v) == 1 for v in wins.values()),
          "EG8 holds one window per dataset across all systems and both arms",
          f"got { {k: sorted(v) for k, v in wins.items()} }")

    dl = {(r["detector"], r["dataset"]): r for r in read(d / "delta_native_vs_aligned.csv")}
    stale = []
    for (det, ds), r in dl.items():
        for col, proto in (("ap_native", "native"), ("ap_aligned", "aligned")):
            if r[col] and not close(float(r[col]), ap(proto, det, ds), 1e-6):
                stale.append(f"{det}/{ds}:{col}")
    check(not stale, "delta_native_vs_aligned.csv agrees with raw_results.csv",
          f"stale: {stale}")


def verify_cost_figure(code: Path) -> None:
    print("\nFigure 9  shared-contract cost")
    d = newest(code / "EG8/results")
    raw = [r for r in read(d / "raw_results.csv") if r["protocol"] == "aligned"]
    best = defaultdict(lambda: [0.0, 0.0])
    for r in raw:
        try:
            t, m = float(r["total_wall_time_sec"]), float(r["run_peak_rss_gib"])
        except ValueError:
            continue
        b = best[r["detector"]]
        b[0], b[1] = max(b[0], t), max(b[1], m)

    exp = {"argus": (124.9, 14.9), "euler": (424.7, 35.0),
           "vgrnn": (392.2, 33.9), "pikachu": (61611.5, 67.7), "anomal": (5528.2, 27.5)}
    ok = True
    for det, (t, gb) in exp.items():
        gt, gm = best[det][0], best[det][1] * 1.073741824
        if not (close(gt, t, 0.2) and close(gm, gb, 0.1)):
            ok = False
            print(f"      {det}: {gt:.1f}s/{gm:.1f}GB vs figure {t}s/{gb}GB")
    check(ok, "all five studied systems are the aligned-protocol maximum")

    lite_t = 107.2
    ratio = best["pikachu"][0] / lite_t
    check(close(ratio, 575, 0.6), "Pikachu / GIDS-Lite runtime ratio = 575x",
          f"got {ratio:.1f}x")


def verify_job_total(code: Path) -> None:
    print("\nCount macros")
    total = 0
    parts = {}
    for sc in ["EG1", "EG2", "EG3", "EG4", "EG5", "EG6", "EG7", "EG8", "EG9"]:
        base = code / sc / "results"
        if not base.exists():
            continue
        for d in sorted(x for x in base.iterdir() if x.is_dir()):
            f = d / "raw_results.csv"
            if f.exists() and not d.name.startswith("2026-03-25"):
                if sc == "EG6" and d.name != "2026-08-04-final":
                    continue
                n = len(read(f))
                parts[f"{sc}/{d.name}"] = n
                total += n
    check(total == 1370, "\\numablation = 1,370 distinct result rows",
          f"got {total}: {parts}")


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--code-root", default=str(Path(__file__).resolve().parent.parent))
    args = ap_.parse_args()
    code = Path(args.code_root)
    print(f"re-deriving paper numbers from {code}")

    for fn in (verify_eg1, verify_eg5, verify_eg6, verify_eg7, verify_detection_cic,
               verify_window, verify_eg8, verify_cost_figure, verify_job_total):
        try:
            fn(code)
        except Exception as exc:
            check(False, f"{fn.__name__} raised", repr(exc))

    passed = sum(1 for ok, _, _ in results if ok)
    print(f"\n{passed}/{len(results)} checks passed")
    for ok, name, detail in results:
        if not ok:
            print(f"  FAIL  {name}  {detail}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
