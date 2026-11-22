# Hipotezler (PROTOCOL_v29.md) ve betimsel karşılaştırmalar. Yüzde fark = 100 x (ilk - ikinci) / ikinci; negatif: ilk
# kol daha kısa. H1 ve H2, 18 sentetik örnekte (20.000 ve 50.000 düğüm) kesin iki yönlü işaret testiyle, iki test
# üzerinde Holm düzeltmesiyle sınanır. TSPLIB örnekleri ve geri kalan her şey betimseldir.
HYPOTHESES = [
    ("H1", STAGED_ARM, LKHMERGE_ARM, "Kademeli birleştirme / LKH'nin kendi tur birleştirmesi"),
    ("H2", STAGED_ARM, DUAL_ARM, "Kademeli birleştirme / iki GA-EAX süreci (run v26'nın birincil kolu)"),
]
DESCRIPTIVE = [
    ("D1", STAGED_ARM, "lkh6", "Kademeli birleştirme / altı LKH süreci"),
    ("D2", STAGED_ARM, TRIALS_ARM, "Kademeli birleştirme / 102 turda tek GA-EAX süreci"),
    ("D3", DUAL_ARM, LKHMERGE_ARM, "İki GA-EAX süreci / LKH'nin kendi tur birleştirmesi"),
    ("D4", TRIALS_ARM, BEST6_ARM, "GA-EAX: 102 tur / altı en iyi tur"),
    ("D5", DUAL_ARM, "lkh6", "İki GA-EAX süreci / altı LKH süreci"),
    ("D6", LKHMERGE_ARM, "lkh6", "LKH'nin kendi tur birleştirmesi / altı LKH süreci"),
]
COMPARISONS = HYPOTHESES + DESCRIPTIVE


def pct(a, b):
    return 100.0 * (a - b) / b


def holm(pvalues):
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def load_finished():
    by_name = {x["name"]: x for x in INSTANCES}
    found = []
    for path in sorted((OUTPUT_ROOT / "instances").glob("*.json")):
        item = json.loads(path.read_text())
        if item.get("complete") and item["name"] in by_name:
            assert set(item["results"]) == set(ARM_NAMES) and item["xy_sha256"] == xy_hash(by_name[item["name"]])
            found.append(item)
    return sorted(found, key=lambda r: r["ordinal"])


records = load_finished()
rows = []
for item in records:
    s = item["staged"]
    converged = lambda e: None if e is None else bool(e["converged"])
    row = dict(instance=item["name"], group=item["group"], family=item["family"], n=item["n"], budget_s=item["budget_s"],
               optimum=item["optimum"], ordinal=item["ordinal"], order="-".join(item["order"]),
               cpu_model=item["hardware"]["cpu_model"], gpu=item["hardware"]["gpu"], session=item["session_id"][:8],
               best_at_cut=s["best_at_cut"], staged_source=s["source"], staged_budget_s=s["budget_s"],
               staged_used_s=s["used_s"], staged_stage1_wall_s=s["stage1_wall_s"],
               staged_stage1_converged=sum(bool(converged(g["eax"])) for g in s["stage1"]),
               staged_stage1_ran=sum(g["eax"] is not None for g in s["stage1"]),
               staged_stage2_converged=converged(s["stage2"]["eax"]), staged_stage2_skipped=s["stage2"]["skipped"])
    for arm in ARM_NAMES:
        row[arm] = item["results"][arm]
        row[arm + "_gap_pct"] = pct(item["results"][arm], item["optimum"]) if item["optimum"] else None
    for arm, _ in EAX_ARMS:
        row[arm + "_converged"] = converged(item["dual"]["arms"][arm]["eax"])
    for key, first, second, _ in COMPARISONS:
        row[key] = pct(item["results"][first], item["results"][second])
    rows.append(row)
frame = pd.DataFrame(rows)
summary_dir = OUTPUT_ROOT / "summary"
frame.to_csv(summary_dir / "per_instance.csv", index=False)

sizes = sorted({n for _, n, *_ in SYNTHETIC})
groups = [("synthetic", DECISION_GROUPS)] + [(f"size:{n // 1000}k", [g for g, m, *_ in SYNTHETIC if m == n]) for n in sizes] + \
         [("class:" + g, [g]) for g, *_ in SYNTHETIC] + [("class:" + g, [g]) for g, _ in TSPLIB_SETS] + \
         [("all", [g for g, *_ in SYNTHETIC] + [g for g, _ in TSPLIB_SETS])]
stats_rows = []
for label, members in groups:
    part = frame[frame["group"].isin(members)] if len(frame) else frame
    if not len(part):
        continue
    for key, first, second, text in COMPARISONS:
        st = B.paired_stats(part[key].to_numpy(), bootstrap_seed=29000 + len(stats_rows))
        stats_rows.append(dict(group=label, key=key, comparison=text, first=first, second=second, **st))
stats = pd.DataFrame(stats_rows)
expected = sum(c for g, _, c, *_ in SYNTHETIC if g in DECISION_GROUPS)
decision_part = stats[stats["group"] == "synthetic"] if len(stats) else stats
complete = len(frame[frame["group"].isin(DECISION_GROUPS)]) == expected if len(frame) else False
tests = {}
if len(decision_part):
    found = [decision_part[decision_part["key"] == key].iloc[0] for key, *_ in HYPOTHESES]
    adjusted = holm([float(r["sign_p"]) for r in found])
    for (key, *_), r, p_adj in zip(HYPOTHESES, found, adjusted):
        tests[key] = dict(win=int(r["win"]), tie=int(r["tie"]), loss=int(r["loss"]), sign_p=float(r["sign_p"]),
                          holm_p=float(p_adj), supported=bool(complete and r["win"] > r["loss"] and p_adj < 0.05))
stats["holm_p"] = [tests[r["key"]]["holm_p"] if r["group"] == "synthetic" and r["key"] in tests else None
                   for _, r in stats.iterrows()] if len(stats) else []
stats.to_csv(summary_dir / "paired_stats.csv", index=False)

timing = {}
for label, members in groups:
    part = frame[frame["group"].isin(members)] if len(frame) else frame
    if len(part):
        timing[label] = dict(instances=int(len(part)),
                             used_share_median=float((part["staged_used_s"] / part["staged_budget_s"]).median()),
                             stage1_wall_median_s=float(part["staged_stage1_wall_s"].median()),
                             stage1_converged=int(part["staged_stage1_converged"].sum()),
                             stage1_ran=int(part["staged_stage1_ran"].sum()),
                             stage2_converged=int(part["staged_stage2_converged"].fillna(False).sum()),
                             source={k: int(v) for k, v in part["staged_source"].value_counts().items()},
                             trials_converged=int(part[TRIALS_ARM + "_converged"].fillna(False).sum()))
gaps = {}
for label, members in [("class:" + g, [g]) for g, _ in TSPLIB_SETS]:
    part = frame[frame["group"].isin(members)] if len(frame) else frame
    if len(part) and part["optimum"].notna().all():
        gaps[label] = {arm: float(part[arm + "_gap_pct"].mean()) for arm in ARM_NAMES}
main = dict(experiment=EXPERIMENT_NAME, instances=len(frame), decision_instances_expected=expected,
            decision_complete=bool(complete), tests=tests,
            staged_merging_confirmed=bool(all(t["supported"] for t in tests.values())) if complete and tests else None,
            timing=timing, tsplib_mean_gap_pct=gaps, test_mode=TEST_MODE, protocol=PROTOCOL, selfcheck=SELFCHECK,
            generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
write_json_atomic(summary_dir / "main.json", main)


def fmt(r):
    return (f"{int(r['win'])} / {int(r['tie'])} / {int(r['loss'])} | {r['mean_delta_pct']:+.4f} | "
            f"[{r['ci95_low_pct']:+.4f}, {r['ci95_high_pct']:+.4f}] | {r['sign_p']:.3g}")


lines = [f"# Run v29 (doğrulama koşusu): kademeli birleştirme — {EXPERIMENT_NAME}", "",
         f"Tamamlanan örnek: {len(frame)} / {len(INSTANCES)} (hipotezler için {expected} sentetik örnek gerekir; tamam: {complete}).", "",
         "## Hipotezler (sentetik örnekler; ilk kol daha kısa / eşit / daha uzun | ortalama fark % | %95 aralık | işaret testi p | Holm p)", ""]
for key, first, second, text in HYPOTHESES:
    row = decision_part[decision_part["key"] == key] if len(decision_part) else decision_part
    t = tests.get(key)
    lines.append(f"- {key} {text}: " + (fmt(row.iloc[0]) + f" | {t['holm_p']:.3g} — destekleniyor mu: {t['supported']}"
                                       if len(row) and t else "veri yok"))
lines += ["", f"Kademeli birleştirme doğrulandı mı (H1 ve H2): **{main['staged_merging_confirmed']}**.", "",
          "## Betimsel karşılaştırmalar (sınıflara göre)", ""]
for label, members in groups[1:]:
    part = stats[stats["group"] == label] if len(stats) else stats
    if not len(part):
        continue
    lines.append(f"### {label}")
    for key, first, second, text in COMPARISONS:
        row = part[part["key"] == key]
        lines.append(f"- {key} {text}: " + (fmt(row.iloc[0]) if len(row) else "veri yok"))
    lines.append("")
lines += ["## TSPLIB: optimuma uzaklık (%)", ""]
for label, g in gaps.items():
    lines.append(f"- {label}: " + ", ".join(f"{arm} {v:.4f}" for arm, v in g.items()))
lines += ["", "## Kademeli kolun süresi", ""]
for label, t in timing.items():
    lines.append(f"- {label}: {t['instances']} örnek; kullanılan süre / kalan süre ortanca {t['used_share_median']:.2f}; "
                 f"aşama 1 ortanca {t['stage1_wall_median_s']:.0f} s, yakınsayan grup {t['stage1_converged']}/{t['stage1_ran']}; "
                 f"aşama 2 yakınsadı: {t['stage2_converged']}; sonucun kaynağı {t['source']}; 102 turda tek GA-EAX "
                 f"yakınsadı: {t['trials_converged']}.")
(summary_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
