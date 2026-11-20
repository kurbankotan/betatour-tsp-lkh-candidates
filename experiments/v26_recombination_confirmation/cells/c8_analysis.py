# Doğrulayıcı testler ve ikincil karşılaştırmalar (PROTOCOL_v26.md). Yüzde fark = 100 x (ilk - ikinci) / ikinci;
# negatif: ilk kol daha kısa. "eax_scratch" bir kol değil referanstır: tek süreç, W x bütçe saniye.
def size_class(item):
    return "tsplib" if item["family"] == "tsplib" else f"n{round(item['n'] / 1000)}k"


SYNTHETIC_CLASSES = sorted({f"n{round(n / 1000)}k" for _, n, *_ in SYNTHETIC}, key=lambda c: int(c[1:-1]))
SMALL_CLASS = f"n{round(min(n for _, n, *_ in SYNTHETIC) / 1000)}k"
# (anahtar, ilk kol, ikinci kol, örnek kümesi, açıklama). Holm düzeltmesi bu dört test üzerinden yapılır.
CONFIRMATORY = [
    ("H1", DUAL_ARM, "lkh6", "synthetic", "İki GA-EAX süreci / altı LKH süreci"),
    ("H2", DUAL_ARM, LKHMERGE_ARM, "synthetic", "İki GA-EAX süreci / LKH'nin kendi tur birleştirmesi"),
    ("H3", DUAL_ARM, "cascade", "synthetic", "İki GA-EAX süreci / bilinen melez"),
    ("H4", TRIALS_ARM, BEST6_ARM, "class:" + SMALL_CLASS, "Atılan deneme turlarının katkısı: 102 tur / altı en iyi tur"),
]
SECONDARY = [
    ("S1", TRIALS_ARM, "lkh6", "102 tur + GA-EAX / altı LKH süreci"),
    ("S2", BEST6_ARM, "lkh6", "Altı en iyi tur + GA-EAX / altı LKH süreci"),
    ("S3", LKHMERGE_ARM, "lkh6", "LKH'nin kendi tur birleştirmesi / altı LKH süreci"),
    ("S4", "cascade", "lkh6", "Bilinen melez / altı LKH süreci"),
    ("S5", BEST6_ARM, LKHMERGE_ARM, "Aynı altı tur: GA-EAX / LKH'nin kendi birleştirmesi"),
    ("R1", "eax_scratch", "lkh6", "Sıfırdan GA-EAX (eşit toplam CPU süresi) / altı LKH süreci"),
    ("R2", DUAL_ARM, "eax_scratch", "İki GA-EAX süreci / sıfırdan GA-EAX"),
]
COMPARISONS = [(k, a, b, label) for k, a, b, _, label in CONFIRMATORY] + SECONDARY


def pct(a, b):
    return 100.0 * (a - b) / b


def flag(value):
    return float("nan") if value is None else float(bool(value))


def load_finished():
    by_name = {x["name"]: x for x in INSTANCES}
    found = []
    for path in sorted((OUTPUT_ROOT / "instances").glob("*.json")):
        item = json.loads(path.read_text())
        if item.get("complete") and item["name"] in by_name:
            assert set(item["results"]) == set(ARM_NAMES) and item["xy_sha256"] == xy_hash(by_name[item["name"]])
            ref_path = OUTPUT_ROOT / "reference" / (item["name"] + ".json")
            item["reference"] = json.loads(ref_path.read_text()) if ref_path.exists() else None
            found.append(item)
    return sorted(found, key=lambda r: r["ordinal"])


def reach_share(item):
    """Birincil kolun, altı LKH sürecinin bütçe sonundaki uzunluğuna ulaştığı an (bütçenin kesri); ulaşmadıysa None."""
    target, dual = item["results"]["lkh6"], item["dual"]
    cut = dual["arms"][TRIALS_ARM]["cut_s"]
    before = [t for run in item["lkh6"]["runs"] for t, c in run["improvements"] if c <= target and t <= cut + 1e-9]
    if before:
        return min(before) / item["budget_s"]
    times = []
    for arm, _ in EAX_ARMS:
        a = dual["arms"][arm]
        if a["eax"] is not None:
            times += [cut + a["prep_s"] + t for t, c in a["eax"]["curve"] if c <= target and t <= a["eax_budget_s"] + 1e-9]
    return min(times) / item["budget_s"] if times else None


records = load_finished()
rows = []
for item in records:
    ref, dual, merge, cascade = item["reference"], item["dual"], item["lkhmerge"], item["cascade"]
    trials, best6 = dual["arms"][TRIALS_ARM], dual["arms"][BEST6_ARM]
    values = dict(item["results"])
    values["eax_scratch"] = ref["at_limit"] if ref and ref["at_limit"] is not None else None
    n, budget = item["n"], item["budget_s"]
    row = dict(instance=item["name"], group=item["group"], family=item["family"], size_class=size_class(item),
               synthetic=item["family"] != "tsplib", n=n, budget_s=budget, optimum=item["optimum"],
               ordinal=item["ordinal"], dual_first=item["dual_first"], lkh_first=item["lkh_first"],
               cpu_model=item["hardware"]["cpu_model"], gpu=item["hardware"]["gpu"], session=item["session_id"][:8])
    for arm in ARM_NAMES + ["eax_scratch"]:
        row[arm] = values[arm]
        if item["optimum"] and values[arm] is not None:
            row[arm + "_gap_pct"] = pct(values[arm], item["optimum"])
    for key, first, second, _ in COMPARISONS:
        a, b = values[first], values[second]
        row[key] = pct(a, b) if a is not None and b is not None else float("nan")
    reached = reach_share(item)
    merge_trials = [r["trials"] for r in merge["runs"]]
    row.update(
        lkh6_mean6=item["lkh6"]["mean_at_budget"], lkh6_trials=item["lkh6"]["mean_trials_at_fraction"][-1],
        lkh6_stopped_from_outside=int(sum(r["stopped_from_outside"] for r in item["lkh6"]["runs"])),
        lkh6_latest_first_tour_s=max(r["first_tour_s"] for r in item["lkh6"]["runs"]),
        lkh6_after_cut_pct=pct(dual["best_at_cut"], values["lkh6"]),
        trials_at_cut=float(np.mean([p["trials_before_cut"] for p in trials["per_process"]])),
        trials_pop=trials["pop_size"], trials_union_per_node=trials["population"]["union_edges"] / n,
        trials_common_per_node=trials["population"]["common_edges"] / n,
        trials_pairwise_shared=trials["population"]["pairwise_shared"],
        best6_distinct=best6["population"]["distinct"], best6_union_per_node=best6["population"]["union_edges"] / n,
        trials_skipped=float(trials["skipped"]), best6_skipped=float(best6["skipped"]),
        trials_converged=flag(None if trials["eax"] is None else trials["eax"]["converged"]),
        best6_converged=flag(None if best6["eax"] is None else best6["eax"]["converged"]),
        trials_eax_budget_s=trials["eax_budget_s"],
        trials_eax_wall_s=float("nan") if trials["eax"] is None else trials["eax"]["wall_s"],
        best6_eax_wall_s=float("nan") if best6["eax"] is None else best6["eax"]["wall_s"],
        dual_source="+".join(sorted(dual["source"])),
        dual_reach_share=float("nan") if reached is None else reached,
        merge_skipped=float(merge["skipped"]), merge_distinct=merge["distinct"],
        merge_common_per_node=merge["common_edges"] / n, merge_budget_s=merge["lkh_budget_s"],
        merge_processes_with_tour=merge["processes_with_tour"] if not merge["skipped"] else float("nan"),
        merge_keeps_common=flag(merge["keeps_common_edges"]),
        merge_trials=float(np.mean(merge_trials)) if merge_trials else float("nan"),
        merge_gain_pct=pct(merge["at_deadline"], merge["best_at_cut"]) if merge["at_deadline"] is not None else float("nan"),
        cascade_tours=cascade["tours"], cascade_generation_s=cascade["generation_s"],
        cascade_converged=flag(None if cascade["eax"] is None else cascade["eax"]["converged"]),
        reference_converged=flag(None if ref is None else ref["eax"]["converged"]),
        reference_init_s=float("nan") if ref is None else ref["eax"]["first_line_s"],
        process_seconds_dual=(W * trials["cut_s"] + sum(a["eax"]["wall_s"] for a in (trials, best6) if a["eax"] is not None))
        / (W * budget),
        process_seconds_merge=(W * merge["cut_s"] + sum(min(r["wall_s"], merge["lkh_budget_s"]) for r in merge["runs"]))
        / (W * budget))
    rows.append(row)

per_instance = pd.DataFrame(rows)
per_instance.to_csv(OUTPUT_ROOT / "summary" / "per_instance.csv", index=False)

stat_rows = []
if len(per_instance):
    frames = [("all", per_instance), ("synthetic", per_instance[per_instance["synthetic"]])] \
        + [("class:" + k, f) for k, f in per_instance.groupby("size_class", sort=True)] \
        + [("group:" + k, f) for k, f in per_instance.groupby("group", sort=True)]
    for group, frame in frames:
        for ci, (key, first, second, label) in enumerate(COMPARISONS):
            values = frame[key].dropna().to_numpy() if len(frame) else np.zeros(0)
            if len(values):
                stat_rows.append(dict(group=group, key=key, comparison=label, first=first, second=second,
                                      **B.paired_stats(values, 20261121 + 100 * ci)))
stats = pd.DataFrame(stat_rows)
stats.to_csv(OUTPUT_ROOT / "summary" / "paired_stats.csv", index=False)

synthetic_names = [x["name"] for x in INSTANCES if x["family"] != "tsplib"]
synthetic_done = set(synthetic_names) <= {r["name"] for r in records}
suite_complete = len(INSTANCES) > 0 and len(records) == len(INSTANCES)
if TEST_MODE:
    note = "TEKNİK ÖZ SINAMA: bu sonuç hiçbir karar için kullanılmaz."
elif not suite_complete:
    note = f"ARA SONUÇ: {len(records)}/{len(INSTANCES)} örnek tamamlandı. Testler, koşu bitince değerlendirilir."
else:
    note = f"{len(records)}/{len(INSTANCES)} örnek tamamlandı."


def stat(group, key):
    hit = stats[(stats["group"] == group) & (stats["key"] == key)] if len(stats) else stats
    return hit.iloc[0] if len(hit) else None


def describe(row):
    return (f"{row['win']}/{row['tie']}/{row['loss']} (ilk kol kısa/eşit/uzun), ortalama %{row['mean_delta_pct']:+.4f} "
            f"(%95 GA {row['ci95_low_pct']:+.4f} ile {row['ci95_high_pct']:+.4f}), ortanca %{row['median_delta_pct']:+.4f}, "
            f"işaret testi p = {row['sign_p']:.4f}")


# Önceden yazılan testler (PROTOCOL_v26.md): dört test, Holm düzeltmesi, alfa = 0,05. Bir hipotez, ilk kol daha çok
# örnekte kısaysa ve düzeltilmiş p < 0,05 ise desteklenir. Testler ancak koşu tamamlanınca değerlendirilir.
confirmatory = {}
if suite_complete and not TEST_MODE:
    found = [stat(group, key) for key, _, _, group, _ in CONFIRMATORY]
    assert all(x is not None for x in found), "doğrulayıcı testlerden birinin verisi yok"
    adjusted = B.holm([float(x["sign_p"]) for x in found])
    for (key, first, second, group, label), row, p_adj in zip(CONFIRMATORY, found, adjusted):
        confirmatory[key] = dict(first=first, second=second, instances_group=group, label=label,
                                 instances=int(row["instances"]), win=int(row["win"]), tie=int(row["tie"]),
                                 loss=int(row["loss"]), mean_delta_pct=float(row["mean_delta_pct"]),
                                 median_delta_pct=float(row["median_delta_pct"]),
                                 ci95_low_pct=float(row["ci95_low_pct"]), ci95_high_pct=float(row["ci95_high_pct"]),
                                 sign_p=float(row["sign_p"]), holm_p=float(p_adj),
                                 supported=bool(row["win"] > row["loss"] and p_adj < 0.05))

classes = SYNTHETIC_CLASSES + (["tsplib"] if len(per_instance) and (~per_instance["synthetic"]).any() else [])
tsplib_groups = [g for g, _ in TSPLIB_SETS]
lines = ["# BetaTour run v26: doğrulama koşusu (LKH-3 turlarının GA-EAX ile birleştirilmesi)", "", note, "",
         f"Birincil kol: `{DUAL_ARM}`. Altı LKH süreci bütçenin %{100 * SPLIT:.0f} anında durur; iki GA-EAX süreci aynı "
         f"anda koşar: biri her sürecin en iyi turu ve son {PER_PROCESS - 1} atılan deneme turuyla (`{TRIALS_ARM}`), "
         f"öteki altı en iyi turla (`{BEST6_ARM}`). Sonuç ikisinden kısa olanıdır. Karşılaştırmalar: `lkh6` (aynı altı "
         f"süreç bütçenin sonuna kadar), `{LKHMERGE_ARM}` (altı en iyi tur LKH'nin kendi tur birleştirmesine verilir) ve "
         "`cascade` (bilinen melez). Bütün zamanlar duvar saatidir.", ""]
lines += ["## Doğrulayıcı testler (koşudan önce yazıldı)", ""]
if confirmatory:
    lines += ["| Test | Karşılaştırma | Örnekler | Kısa/eşit/uzun | Ortalama fark (%) | %95 GA | Ortanca (%) | p | Holm p | Destek |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for key, c in confirmatory.items():
        lines.append(f"| {key} | `{c['first']}` / `{c['second']}` | {c['instances_group'].replace('class:', '')} "
                     f"({c['instances']}) | {c['win']}/{c['tie']}/{c['loss']} | {c['mean_delta_pct']:+.4f} | "
                     f"{c['ci95_low_pct']:+.4f} ile {c['ci95_high_pct']:+.4f} | {c['median_delta_pct']:+.4f} | "
                     f"{c['sign_p']:.2e} | {c['holm_p']:.2e} | **{'evet' if c['supported'] else 'hayır'}** |")
else:
    lines.append("Koşu tamamlanınca değerlendirilir." if not TEST_MODE else "Öz sınamada değerlendirilmez.")
lines += ["", "## Bütçe sonunda karşılaştırmalar (doğrulayıcı testler dışında hepsi keşif amaçlıdır)", "",
          "| Grup | Karşılaştırma | Örnek | Kısa/eşit/uzun | Ortalama fark (%) | %95 GA | Ortanca (%) | p |",
          "|---|---|---|---|---|---|---|---|"]
for group in ["synthetic"] + ["class:" + c for c in SYNTHETIC_CLASSES] + ["group:" + g for g in tsplib_groups]:
    for key, first, second, label in COMPARISONS:
        row = stat(group, key)
        if row is not None:
            lines.append(f"| {group.split(':')[-1]} | {key}: `{first}` / `{second}` | {row['instances']} | "
                         f"{row['win']}/{row['tie']}/{row['loss']} | {row['mean_delta_pct']:+.4f} | "
                         f"{row['ci95_low_pct']:+.4f} ile {row['ci95_high_pct']:+.4f} | {row['median_delta_pct']:+.4f} | "
                         f"{row['sign_p']:.4f} |")
shown = ["lkh6", BEST6_ARM, TRIALS_ARM, DUAL_ARM, LKHMERGE_ARM, "cascade", "eax_scratch"]
for group in tsplib_groups:
    sub = per_instance[per_instance["group"] == group] if len(per_instance) else per_instance
    if not len(sub):
        continue
    lines += ["", f"## Optimuma uzaklık (%): {group}", "",
              "| Örnek | Düğüm | Bütçe (s) | " + " | ".join(f"`{a}`" for a in shown) + " |", "|---|---|---|" + "---|" * len(shown)]
    for _, r in sub.iterrows():
        lines.append(f"| {r['instance']} | {r['n']} | {r['budget_s']:.0f} | "
                     + " | ".join("-" if pd.isna(r.get(a + "_gap_pct", float("nan"))) else f"{r[a + '_gap_pct']:.4f}"
                                  for a in shown) + " |")
    lines.append("| ortalama | | | " + " | ".join(
        "-" if (a + "_gap_pct") not in sub or sub[a + "_gap_pct"].isna().any() else f"{sub[a + '_gap_pct'].mean():.4f}"
        for a in shown) + " |")
if len(per_instance):
    lines += ["", "## Mekanizma (sınıf ortalamaları)", "",
              "| Sınıf | Örnek | Kesimde deneme/süreç | 102 turun birleşim kenarı/düğüm | GA-EAX yakınsadı (102 tur / 6 tur) | "
              "Birincil kolun kaynağı | LKH son uzunluğuna ulaşma (bütçenin kesri) | LKH birleştirmesi: turu olan süreç, "
              "kesimdeki tura göre kazanç (%) | Süreç-saniye payı (birincil / LKH birleştirmesi) |",
              "|---|---|---|---|---|---|---|---|---|"]
    for c in classes:
        f = per_instance[per_instance["size_class"] == c]
        share = f["dual_reach_share"].dropna()
        source = ", ".join(f"{k} {v}" for k, v in f["dual_source"].value_counts().sort_index().items())
        lines.append(f"| {c} | {len(f)} | {f['trials_at_cut'].mean():.0f} | {f['trials_union_per_node'].mean():.3f} | "
                     f"{int(f['trials_converged'].sum())}/{int(f['trials_converged'].notna().sum())} / "
                     f"{int(f['best6_converged'].sum())}/{int(f['best6_converged'].notna().sum())} | {source} | "
                     + (f"{share.mean():.2f} ({len(share)}/{len(f)} örnek)" if len(share) else "ulaşmadı")
                     + f" | {f['merge_processes_with_tour'].mean():.1f}, {f['merge_gain_pct'].mean():+.4f} | "
                     f"{f['process_seconds_dual'].mean():.2f} / {f['process_seconds_merge'].mean():.2f} |")
flags = []
if len(per_instance):
    early = per_instance[per_instance["lkh6_stopped_from_outside"] < W]
    if len(early):
        flags.append("Bazı LKH süreçleri bütçeden önce kendiliğinden bitti: " + ", ".join(early["instance"]))
    if per_instance["session"].nunique() > 1 or per_instance["cpu_model"].nunique() > 1:
        flags.append(f"Koşu {per_instance['session'].nunique()} oturumda, {per_instance['cpu_model'].nunique()} işlemci modelinde yapıldı.")
    few = per_instance[(per_instance["merge_skipped"] == 0) & (per_instance["merge_processes_with_tour"] < W)]
    if len(few):
        flags.append("LKH birleştirmesinde süresi içinde tur kaydetmeyen süreç var: " + ", ".join(few["instance"]))
    broken = per_instance[per_instance["merge_keeps_common"] == 0]
    if len(broken):
        flags.append("LKH birleştirmesinin turu ortak kenarların hepsini taşımıyor: " + ", ".join(broken["instance"]))
    skipped = per_instance[(per_instance["trials_skipped"] + per_instance["best6_skipped"] + per_instance["merge_skipped"]) > 0]
    if len(skipped):
        flags.append("Farklı turu ikiden az olan popülasyon (birleştirme koşmadı, sonuç kesimdeki tur): "
                     + ", ".join(skipped["instance"]))
lines += ["", "## Uyarılar", ""] + (["- " + x for x in flags] if flags else ["Yok."])
lines += ["", "Doğrulayıcı testler yalnızca dört testtir (H1-H4) ve yalnızca yeni sentetik örnekleri kullanır; öteki "
          "bütün karşılaştırmalar keşif amaçlıdır ve düzeltilmemiştir. GA-EAX tek çekirdekte koşar: birincil kol kesimden "
          "sonra iki çekirdek kullanır, `lkh6` ve LKH'nin kendi birleştirmesi altı çekirdek kullanır.", ""]
report = "\n".join(lines)
(OUTPUT_ROOT / "summary" / "report.md").write_text(report, encoding="utf-8")
write_json_atomic(OUTPUT_ROOT / "summary" / "main.json",
                  dict(note=note, suite_complete=bool(suite_complete), synthetic_complete=bool(synthetic_done),
                       instances=len(records), total=len(INSTANCES), primary_arm=DUAL_ARM,
                       confirmatory=confirmatory, reference_done=int(sum(r["reference"] is not None for r in records))))
print(report)
