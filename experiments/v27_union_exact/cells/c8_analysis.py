# Okumalar (karar kuralı, PROTOCOL_v27.md) ve betimsel karşılaştırmalar. Yüzde fark = 100 x (ilk - ikinci) / ikinci;
# negatif: ilk kol daha kısa. Okumalar 12 örnekte (10.000 düğüm); 20.000 düğüm yalnızca betimseldir. Pilotta
# düzeltme yoktur.
READINGS = [
    ("R1", EXACT_ARM, BEST6_ARM, "Aynı altı tur: birleşimde tam çözüm / GA-EAX"),
    ("R2", EXACT_ARM, DUAL_ARM, "Birleşimde tam çözüm / iki GA-EAX süreci (run v26'nın birincil kolu)"),
    ("R3", EXACT_BETA_ARM, EXACT_ARM, "Birleşim + BetaTour kenarları / birleşim"),
    ("R4", EXACT_BETA_ARM, EXACT_NEAR_ARM, "Birleşim + BetaTour kenarları / birleşim + en kısa kenarlar"),
]
DESCRIPTIVE = [
    ("D1", EXACT_ARM, LKHMERGE_ARM, "Birleşimde tam çözüm / LKH'nin kendi tur birleştirmesi"),
    ("D2", EXACT_ARM, "lkh6", "Birleşimde tam çözüm / altı LKH süreci"),
    ("D3", EXACT_NEAR_ARM, EXACT_ARM, "Birleşim + en kısa kenarlar / birleşim"),
    ("D4", DUAL_ARM, LKHMERGE_ARM, "İki GA-EAX süreci / LKH'nin kendi tur birleştirmesi"),
    ("D5", BEST6_ARM, LKHMERGE_ARM, "Aynı altı tur: GA-EAX / LKH'nin kendi birleştirmesi"),
    ("D6", DUAL_ARM, "lkh6", "İki GA-EAX süreci / altı LKH süreci"),
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
    exact = item["exact"]["arms"]
    row = dict(instance=item["name"], group=item["group"], family=item["family"], n=item["n"], budget_s=item["budget_s"],
               ordinal=item["ordinal"], order="-".join(item["order"]), cpu_model=item["hardware"]["cpu_model"],
               gpu=item["hardware"]["gpu"], session=item["session_id"][:8],
               best_at_cut=item["exact"]["best_at_cut"], union_edges=item["exact"]["union_edges"],
               k_extra=item["exact"]["k_extra"], pairwise_shared=item["exact"]["pairwise"]["pairwise_shared"])
    for arm in ARM_NAMES:
        row[arm] = item["results"][arm]
    for arm in TOUR_ARMS:
        row["outside_union6_" + arm] = item["edges_outside_union6"][arm]
    for arm, _ in EXACT_ARMS:
        a = exact[arm]
        row.update({f"{arm}_status": a["solver"]["status"], f"{arm}_optimal": bool(a["solver"]["optimal"]),
                    f"{arm}_solve_s": a["solver"]["solve_s"], f"{arm}_budget_s": a["exact_budget_s"],
                    f"{arm}_core": a["solver"]["core"], f"{arm}_free": a["solver"]["free"],
                    f"{arm}_bnb_nodes": a["solver"]["bnb_nodes"], f"{arm}_gap": a["solver"]["gap"],
                    f"{arm}_extra_used": a["extra_used"], f"{arm}_late": a["best_found_late"],
                    f"{arm}_after_deadline_s": a["finished_after_deadline_s"]})
    for arm, _ in EAX_ARMS:
        e = item["dual"]["arms"][arm]["eax"]
        row[arm + "_converged"] = None if e is None else bool(e["converged"])
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
        s = B.paired_stats(part[key].to_numpy(), bootstrap_seed=27000 + len(stats_rows))
        stats_rows.append(dict(group=label, key=key, comparison=text, first=first, second=second, **s))
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
exact_holds = bool(readings["R1"] or readings["R2"]) if complete else None
model_holds = bool(readings["R3"] and readings["R4"]) if complete else None
solver_rows = []
for arm, _ in EXACT_ARMS:
    for label, members in groups:
        part = frame[frame["group"].isin(members)] if len(frame) else frame
        if not len(part):
            continue
        solver_rows.append(dict(arm=arm, group=label, instances=len(part),
                                optimal=int(part[f"{arm}_optimal"].sum()), median_solve_s=float(part[f"{arm}_solve_s"].median()),
                                max_solve_s=float(part[f"{arm}_solve_s"].max()), median_core=float(part[f"{arm}_core"].median()),
                                extra_used_mean=float(part[f"{arm}_extra_used"].mean()),
                                late=int(part[f"{arm}_late"].sum())))
solver = pd.DataFrame(solver_rows)
solver.to_csv(summary_dir / "exact_solver.csv", index=False)
same_as_eax = int((frame[EXACT_ARM] == frame[BEST6_ARM]).sum()) if len(frame) else 0
main = dict(experiment=EXPERIMENT_NAME, instances=len(frame), decision_instances_expected=expected,
            decision_complete=bool(complete), readings=readings, exact_merging_holds=exact_holds,
            model_edges_holds=model_holds, exact_equal_to_eax_best6=same_as_eax,
            eax_best6_outside_union_mean=float(frame["outside_union6_" + BEST6_ARM].mean()) if len(frame) else None,
            eax_trials_outside_union_mean=float(frame["outside_union6_" + TRIALS_ARM].mean()) if len(frame) else None,
            test_mode=TEST_MODE, protocol=PROTOCOL, selfcheck=SELFCHECK,
            generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
write_json_atomic(summary_dir / "main.json", main)


def fmt(r):
    return (f"{int(r['win'])} / {int(r['tie'])} / {int(r['loss'])} | {r['mean_delta_pct']:+.4f} | "
            f"[{r['ci95_low_pct']:+.4f}, {r['ci95_high_pct']:+.4f}] | {r['sign_p']:.3g}")


lines = [f"# Run v27 (pilot): birleşim grafında tam çözüm — {EXPERIMENT_NAME}", "",
         f"Tamamlanan örnek: {len(frame)} / {len(INSTANCES)} (okumalar için {expected} örnek gerekir; tamam: {complete}).", "",
         "## Okumalar (10.000 düğüm, 12 örnek; ilk kol daha kısa / eşit / daha uzun | ortalama fark % | %95 aralık | işaret testi p)", ""]
for key, first, second, text in READINGS:
    row = decision_part[decision_part["key"] == key]
    lines.append(f"- {key} {text}: " + (fmt(row.iloc[0]) if len(row) else "veri yok") + f" — tutar mı: {readings[key]}")
lines += ["", f"Tam çözüm GA-EAX'ten kısa mı (R1 ya da R2): **{exact_holds}**. Model kenarları katkı veriyor mu (R3 ve R4): "
          f"**{model_holds}**.", "", "## Betimsel karşılaştırmalar (bütün örnekler)", ""]
for key, first, second, text in DESCRIPTIVE:
    row = stats[(stats["group"] == "all") & (stats["key"] == key)] if len(stats) else stats
    lines.append(f"- {key} {text}: " + (fmt(row.iloc[0]) if len(row) else "veri yok"))
lines += ["", "## Tam çözücü", ""]
for _, r in solver[solver["group"] == "all"].iterrows() if len(solver) else []:
    lines.append(f"- {r['arm']}: {r['optimal']}/{r['instances']} örnekte kanıtlanmış en kısa tur; çözüm süresi ortanca "
                 f"{r['median_solve_s']:.1f} s, en çok {r['max_solve_s']:.1f} s; çekirdek ortanca {r['median_core']:.0f} düğüm; "
                 f"kullanılan ek kenar ortalaması {r['extra_used_mean']:.2f}; süre sonrası bulunan sonuç: {r['late']}.")
lines += [f"- Tam çözüm (altı tur) ile GA-EAX'in (aynı altı tur) aynı uzunlukta olduğu örnek: {same_as_eax}.",
          f"- GA-EAX turlarının altı turun birleşiminde olmayan kenarları (ortalama): altı tur "
          f"{main['eax_best6_outside_union_mean']}, 102 tur {main['eax_trials_outside_union_mean']}.", ""]
(summary_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
