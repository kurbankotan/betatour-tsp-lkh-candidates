# Okumalar (karar kuralı, PROTOCOL_v28.md) ve betimsel karşılaştırmalar. Yüzde fark = 100 x (ilk - ikinci) / ikinci;
# negatif: ilk kol daha kısa. Okumalar 12 örnekte (20.000 düğüm); 50.000 düğüm yalnızca betimseldir. Pilotta
# düzeltme yoktur.
READINGS = [
    ("R1", STAGED_ARM, DUAL_ARM, "Kademeli birleştirme / iki GA-EAX süreci (run v26'nın birincil kolu)"),
    ("R2", STAGED_ARM, LKHMERGE_ARM, "Kademeli birleştirme / LKH'nin kendi tur birleştirmesi"),
]
DESCRIPTIVE = [
    ("D1", STAGED_ARM, "lkh6", "Kademeli birleştirme / altı LKH süreci"),
    ("D2", STAGED_ARM, TRIALS_ARM, "Kademeli birleştirme / 102 turda tek GA-EAX süreci"),
    ("D3", DUAL_ARM, LKHMERGE_ARM, "İki GA-EAX süreci / LKH'nin kendi tur birleştirmesi"),
    ("D4", TRIALS_ARM, BEST6_ARM, "GA-EAX: 102 tur / altı en iyi tur"),
    ("D5", DUAL_ARM, "lkh6", "İki GA-EAX süreci / altı LKH süreci"),
    ("D6", LKHMERGE_ARM, "lkh6", "LKH'nin kendi tur birleştirmesi / altı LKH süreci"),
]
COMPARISONS = READINGS + DESCRIPTIVE


def pct(a, b):
    return 100.0 * (a - b) / b


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
               ordinal=item["ordinal"], order="-".join(item["order"]), cpu_model=item["hardware"]["cpu_model"],
               gpu=item["hardware"]["gpu"], session=item["session_id"][:8], best_at_cut=s["best_at_cut"],
               staged_source=s["source"], staged_budget_s=s["budget_s"], staged_used_s=s["used_s"],
               staged_stage1_wall_s=s["stage1_wall_s"],
               staged_stage1_converged=sum(bool(converged(g["eax"])) for g in s["stage1"]),
               staged_stage1_ran=sum(g["eax"] is not None for g in s["stage1"]),
               staged_stage2_converged=converged(s["stage2"]["eax"]), staged_stage2_skipped=s["stage2"]["skipped"])
    for arm in ARM_NAMES:
        row[arm] = item["results"][arm]
    for arm, _ in EAX_ARMS:
        row[arm + "_converged"] = converged(item["dual"]["arms"][arm]["eax"])
    for key, first, second, _ in COMPARISONS:
        row[key] = pct(item["results"][first], item["results"][second])
    rows.append(row)
frame = pd.DataFrame(rows)
summary_dir = OUTPUT_ROOT / "summary"
frame.to_csv(summary_dir / "per_instance.csv", index=False)

groups = [("decision", DECISION_GROUPS)] + [("class:" + g, [g]) for g, *_ in SYNTHETIC] + \
         [("all", [g for g, *_ in SYNTHETIC])]
stats_rows = []
for label, members in groups:
    part = frame[frame["group"].isin(members)] if len(frame) else frame
    if not len(part):
        continue
    for key, first, second, text in COMPARISONS:
        st = B.paired_stats(part[key].to_numpy(), bootstrap_seed=28000 + len(stats_rows))
        stats_rows.append(dict(group=label, key=key, comparison=text, first=first, second=second, **st))
stats = pd.DataFrame(stats_rows)
stats.to_csv(summary_dir / "paired_stats.csv", index=False)

expected = sum(c for g, _, c, *_ in SYNTHETIC if g in DECISION_GROUPS)
decision_part = stats[stats["group"] == "decision"] if len(stats) else stats
complete = len(frame[frame["group"].isin(DECISION_GROUPS)]) == expected if len(frame) else False


def holds(key):
    row = decision_part[decision_part["key"] == key]
    if not len(row):
        return None
    r = row.iloc[0]
    return bool(r["win"] > r["loss"] and r["sign_p"] < 0.05)


readings = {key: holds(key) for key, *_ in READINGS}
staged_holds = bool(readings["R1"] and readings["R2"]) if complete else None
timing = {}
for label, members in groups:
    part = frame[frame["group"].isin(members)] if len(frame) else frame
    if len(part):
        timing[label] = dict(instances=int(len(part)), used_share_median=float((part["staged_used_s"] / part["staged_budget_s"]).median()),
                             stage1_wall_median_s=float(part["staged_stage1_wall_s"].median()),
                             stage1_converged=int(part["staged_stage1_converged"].sum()),
                             stage1_ran=int(part["staged_stage1_ran"].sum()),
                             stage2_converged=int(part["staged_stage2_converged"].fillna(False).sum()),
                             source={k: int(v) for k, v in part["staged_source"].value_counts().items()},
                             trials_converged=int(part[TRIALS_ARM + "_converged"].fillna(False).sum()))
main = dict(experiment=EXPERIMENT_NAME, instances=len(frame), decision_instances_expected=expected,
            decision_complete=bool(complete), readings=readings, staged_merging_holds=staged_holds, timing=timing,
            test_mode=TEST_MODE, protocol=PROTOCOL, selfcheck=SELFCHECK,
            generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
write_json_atomic(summary_dir / "main.json", main)


def fmt(r):
    return (f"{int(r['win'])} / {int(r['tie'])} / {int(r['loss'])} | {r['mean_delta_pct']:+.4f} | "
            f"[{r['ci95_low_pct']:+.4f}, {r['ci95_high_pct']:+.4f}] | {r['sign_p']:.3g}")


lines = [f"# Run v28 (pilot): kademeli birleştirme — {EXPERIMENT_NAME}", "",
         f"Tamamlanan örnek: {len(frame)} / {len(INSTANCES)} (okumalar için {expected} örnek gerekir; tamam: {complete}).", "",
         "## Okumalar (20.000 düğüm, 12 örnek; ilk kol daha kısa / eşit / daha uzun | ortalama fark % | %95 aralık | işaret testi p)", ""]
for key, first, second, text in READINGS:
    row = decision_part[decision_part["key"] == key]
    lines.append(f"- {key} {text}: " + (fmt(row.iloc[0]) if len(row) else "veri yok") + f" — tutar mı: {readings[key]}")
lines += ["", f"Kademeli birleştirme hem iki GA-EAX sürecinden hem LKH'nin kendi birleştirmesinden kısa mı (R1 ve R2): "
          f"**{staged_holds}**.", "", "## Betimsel karşılaştırmalar (bütün örnekler)", ""]
for key, first, second, text in DESCRIPTIVE:
    row = stats[(stats["group"] == "all") & (stats["key"] == key)] if len(stats) else stats
    lines.append(f"- {key} {text}: " + (fmt(row.iloc[0]) if len(row) else "veri yok"))
lines += ["", "## Kademeli kolun süresi", ""]
for label, t in timing.items():
    lines.append(f"- {label}: {t['instances']} örnek; kullanılan süre / kalan süre ortanca {t['used_share_median']:.2f}; "
                 f"aşama 1 ortanca {t['stage1_wall_median_s']:.0f} s, yakınsayan grup {t['stage1_converged']}/{t['stage1_ran']}; "
                 f"aşama 2 yakınsadı: {t['stage2_converged']}; sonucun kaynağı {t['source']}; 102 turda tek GA-EAX "
                 f"yakınsadı: {t['trials_converged']}.")
(summary_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
