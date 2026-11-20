# (anahtar, ilk kol, ikinci kol, açıklama). Yüzde fark = 100 x (ilk - ikinci) / ikinci; negatif: ilk kol daha kısa.
# "eax_scratch" bir kol değil referanstır: tek süreç, W x bütçe saniye (altı LKH sürecinin toplam süresi).
COMPARISONS = [
    ("P1", PRIMARY_ARM, "lkh6", "Atılan turlar + GA-EAX / altı LKH süreci"),
    ("C1", CONTROL_ARM, "lkh6", "Yalnızca altı en iyi tur + GA-EAX / altı LKH süreci"),
    ("C2", PRIMARY_ARM, CONTROL_ARM, "Atılan turların katkısı: atılan turlar / yalnızca altı en iyi tur"),
    ("C3", "cascade", "lkh6", "Bilinen melez (tek denemelik turlar + GA-EAX) / altı LKH süreci"),
    ("C4", PRIMARY_ARM, "cascade", "Atılan turlar / bilinen melez"),
] + [(f"S{i}", arm, "lkh6", f"Diğer bölme: {arm} / altı LKH süreci")
     for i, arm in enumerate([a for a, _, _ in MERGE_ARMS if a not in (PRIMARY_ARM, CONTROL_ARM)], 1)] + [
    ("R1", "eax_scratch", "lkh6", "Sıfırdan GA-EAX (eşit toplam CPU süresi) / altı LKH süreci"),
    ("R2", PRIMARY_ARM, "eax_scratch", "Atılan turlar / sıfırdan GA-EAX (eşit toplam CPU süresi)"),
]


def pct(a, b):
    return 100.0 * (a - b) / b


def size_class(item):
    return "tsplib" if item["group"] == "tsplib" else f"n{round(item['n'] / 1000)}k"


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


def match_time(item, arm):
    """Kolun, altı LKH sürecinin bütçe sonundaki uzunluğuna ulaştığı an (ortak saat, saniye); ulaşmadıysa None."""
    target, merge = item["results"]["lkh6"], item["merges"][arm]
    if merge["best_at_cut"] <= target:
        return merge["cut_s"]
    for t, cost in merge["eax"]["curve"]:
        if cost <= target and t <= merge["eax_budget_s"] + 1e-9:
            return merge["cut_s"] + merge["prep_s"] + t
    return None


records = load_finished()
rows = []
for item in records:
    primary, cascade, ref = item["merges"][PRIMARY_ARM], item["cascade"], item["reference"]
    values = dict(item["results"])
    values["eax_scratch"] = ref["at_limit"] if ref and ref["at_limit"] is not None else None
    reached = match_time(item, PRIMARY_ARM)
    row = dict(instance=item["name"], group=item["group"], size_class=size_class(item), n=item["n"],
               budget_s=item["budget_s"], optimum=item["optimum"], ordinal=item["ordinal"],
               lkh_first=item["lkh_first"], cpu_model=item["hardware"]["cpu_model"], gpu=item["hardware"]["gpu"],
               session=item["session_id"][:8])
    for arm in ARM_NAMES + ["eax_scratch"]:
        row[arm] = values[arm]
        if item["optimum"] and values[arm] is not None:
            row[arm + "_gap_pct"] = pct(values[arm], item["optimum"])
    for key, first, second, _ in COMPARISONS:
        a, b = values[first], values[second]
        row[key] = pct(a, b) if a is not None and b is not None else float("nan")
    row.update(
        lkh6_mean6=item["lkh6"]["mean_at_budget"], lkh6_trials=item["lkh6"]["mean_trials_at_fraction"][-1],
        lkh6_stopped_from_outside=int(sum(r["stopped_from_outside"] for r in item["lkh6"]["runs"])),
        lkh6_latest_first_tour_s=max(r["first_tour_s"] for r in item["lkh6"]["runs"]),
        primary_pop_size=primary["pop_size"],
        primary_trials_at_cut=float(np.mean([p["trials_before_cut"] for p in primary["per_process"]])),
        primary_union_per_node=primary["population"]["union_edges"] / item["n"],
        primary_common_per_node=primary["population"]["common_edges"] / item["n"],
        primary_pairwise_shared=primary["population"]["pairwise_shared"], primary_prep_s=primary["prep_s"],
        primary_eax_budget_s=primary["eax_budget_s"], primary_eax_wall_s=primary["eax"]["wall_s"],
        primary_eax_generations=primary["eax"]["generations"], primary_eax_converged=primary["eax"]["converged"],
        primary_match_time_s=reached if reached is not None else float("nan"),
        primary_match_time_share=reached / item["budget_s"] if reached is not None else float("nan"),
        cascade_tours=cascade["tours"], cascade_generation_s=cascade["generation_s"],
        cascade_eax_budget_s=cascade["eax_budget_s"], cascade_eax_converged=cascade["eax"]["converged"],
        cascade_pairwise_shared=cascade["population"]["pairwise_shared"],
        reference_converged=ref["eax"]["converged"] if ref else None,
        reference_wall_s=ref["eax"]["wall_s"] if ref else float("nan"),
        reference_init_s=ref["eax"]["first_line_s"] if ref else float("nan"))
    rows.append(row)

per_instance = pd.DataFrame(rows)
per_instance.to_csv(OUTPUT_ROOT / "summary" / "per_instance.csv", index=False)

stat_rows = []
if len(per_instance):
    frames = [("all", per_instance)] + [("class:" + k, f) for k, f in per_instance.groupby("size_class", sort=True)] \
        + [("group:" + k, f) for k, f in per_instance.groupby("group", sort=True)]
    for group, frame in frames:
        for ci, (key, first, second, label) in enumerate(COMPARISONS):
            values = frame[key].dropna().to_numpy()
            if len(values):
                stat_rows.append(dict(group=group, key=key, comparison=label, first=first, second=second,
                                      **B.paired_stats(values, 20261125 + 100 * ci)))
stats = pd.DataFrame(stat_rows)
stats.to_csv(OUTPUT_ROOT / "summary" / "paired_stats.csv", index=False)

suite_complete = len(INSTANCES) > 0 and len(records) == len(INSTANCES)
if TEST_MODE:
    note = "TEKNİK ÖZ SINAMA: bu sonuç hiçbir karar için kullanılmaz."
elif not suite_complete:
    note = f"ARA SONUÇ: {len(records)}/{len(INSTANCES)} örnek tamamlandı. Karar, koşu bitince verilir."
else:
    note = f"{len(records)}/{len(INSTANCES)} örnek tamamlandı."


def stat(group, key):
    hit = stats[(stats["group"] == group) & (stats["key"] == key)] if len(stats) else stats
    return hit.iloc[0] if len(hit) else None


def describe(row):
    return (f"{row['win']}/{row['tie']}/{row['loss']} (ilk kol kısa/eşit/uzun), ortalama %{row['mean_delta_pct']:+.4f} "
            f"(%95 GA {row['ci95_low_pct']:+.4f} ile {row['ci95_high_pct']:+.4f}), ortanca %{row['median_delta_pct']:+.4f}, "
            f"işaret testi p = {row['sign_p']:.4f}")


# Önceden yazılan karar kuralı (PROTOCOL_v25.md). Pilot keşif amaçlıdır; kural yalnızca bir doğrulama koşusunun
# hazırlanıp hazırlanmayacağını belirler.
classes = sorted(per_instance["size_class"].unique(), key=lambda c: (c == "tsplib", c)) if len(per_instance) else []
synthetic_classes = sorted([c for c in classes if c != "tsplib"], key=lambda c: int(c[1:-1]))
small, large = (synthetic_classes[0], synthetic_classes[-1]) if synthetic_classes else (None, None)
decision = {}
if suite_complete and not TEST_MODE and small and large and small != large:
    d1, d2, d3 = stat("class:" + small, "P1"), stat("class:" + large, "P1"), stat("class:" + small, "C2")
    decision["D1_replication_small"] = bool(d1["win"] > d1["loss"] and d1["sign_p"] < 0.05)
    decision["D2_growth_with_size"] = bool(d2["loss"] == 0 and d2["tie"] == 0 and d2["mean_delta_pct"] < d1["mean_delta_pct"])
    decision["D3_discarded_tours_matter"] = bool(d3["win"] > d3["loss"] and d3["sign_p"] < 0.05)
    decision["confirmation_run"] = bool(decision["D1_replication_small"] and decision["D2_growth_with_size"])

lines = ["# BetaTour run v25 (pilot): atılan deneme turlarının birleştirilmesi", "", note, "",
         f"Birincil kol: `{PRIMARY_ARM}` (altı LKH süreci bütçenin yarısında durur; her sürecin en iyi turu ve son "
         f"{PER_PROCESS - 1} atılan deneme turu GA-EAX ile birleştirilir; toplam süre bütçeye eşittir). Karşılaştırma: "
         "`lkh6` (aynı altı süreç bütçenin sonuna kadar sürer; altısının en iyisi). Bütün zamanlar duvar saatidir.", ""]
lines += ["## Bütçe sonunda karşılaştırmalar", "",
          "| Grup | Karşılaştırma | Örnek | Kısa/eşit/uzun | Ortalama fark (%) | %95 GA | Ortanca (%) | p |", "|---|---|---|---|---|---|---|---|"]
for group in ["all"] + ["class:" + c for c in classes]:
    for key, first, second, label in COMPARISONS:
        row = stat(group, key)
        if row is not None:
            lines.append(f"| {group.replace('class:', '')} | {key}: `{first}` / `{second}` | {row['instances']} | "
                         f"{row['win']}/{row['tie']}/{row['loss']} | {row['mean_delta_pct']:+.4f} | "
                         f"{row['ci95_low_pct']:+.4f} ile {row['ci95_high_pct']:+.4f} | {row['median_delta_pct']:+.4f} | "
                         f"{row['sign_p']:.4f} |")
with_optimum = per_instance[per_instance["optimum"].notna()] if len(per_instance) else per_instance
if len(with_optimum):
    shown = ["lkh6", CONTROL_ARM, PRIMARY_ARM, "cascade", "eax_scratch"]
    lines += ["", "## Optimumu bilinen örneklerde optimuma uzaklık (%)", "",
              "| Örnek | Düğüm | Bütçe (s) | " + " | ".join(f"`{a}`" for a in shown) + " |", "|---|---|---|" + "---|" * len(shown)]
    for _, r in with_optimum.iterrows():
        lines.append(f"| {r['instance']} | {r['n']} | {r['budget_s']:.0f} | "
                     + " | ".join("-" if pd.isna(r.get(a + "_gap_pct", float("nan"))) else f"{r[a + '_gap_pct']:.4f}"
                                  for a in shown) + " |")
    lines.append("| ortalama | | | " + " | ".join(
        "-" if with_optimum[a + "_gap_pct"].isna().any() else f"{with_optimum[a + '_gap_pct'].mean():.4f}"
        for a in shown if a + "_gap_pct" in with_optimum) + " |")
if len(per_instance):
    lines += ["", "## Mekanizma (birincil kol, sınıf ortalamaları)", "",
              "| Sınıf | Örnek | Kesimde deneme/süreç | Birleşim kenarı/düğüm | Ortak kenar/düğüm | İkili ortak kenar payı | "
              "GA-EAX süresi (s) | Yakınsayan | LKH'nin son uzunluğuna ulaşma (bütçenin kesri) | Bilinen melezde tur sayısı |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for c in classes:
        f = per_instance[per_instance["size_class"] == c]
        share = f["primary_match_time_share"].dropna()
        lines.append(f"| {c} | {len(f)} | {f['primary_trials_at_cut'].mean():.0f} | {f['primary_union_per_node'].mean():.3f} | "
                     f"{f['primary_common_per_node'].mean():.3f} | {f['primary_pairwise_shared'].mean():.3f} | "
                     f"{f['primary_eax_wall_s'].mean():.0f} | {int(f['primary_eax_converged'].sum())}/{len(f)} | "
                     + (f"{share.mean():.2f} ({len(share)}/{len(f)} örnek)" if len(share) else "ulaşmadı")
                     + f" | {f['cascade_tours'].mean():.0f} |")
lines += ["", "## Karar kuralı (koşudan önce yazıldı)", ""]
if decision:
    lines += [f"- D1, küçük boyutta tekrar ({small}): `{PRIMARY_ARM}` / `lkh6` daha çok örnekte kısa ve p < 0,05: "
              f"**{'evet' if decision['D1_replication_small'] else 'hayır'}** ({describe(stat('class:' + small, 'P1'))}).",
              f"- D2, kazanç boyutla büyüyor ({large}): bütün örneklerde kısa ve ortalama fark {small} sınıfındakinden büyük: "
              f"**{'evet' if decision['D2_growth_with_size'] else 'hayır'}** ({describe(stat('class:' + large, 'P1'))}).",
              f"- D3, atılan turların katkısı ({small}): `{PRIMARY_ARM}` / `{CONTROL_ARM}` daha çok örnekte kısa ve p < 0,05: "
              f"**{'evet' if decision['D3_discarded_tours_matter'] else 'hayır'}** ({describe(stat('class:' + small, 'C2'))}).",
              "", "**Sonuç:** " + ("D1 ve D2 sağlandı: doğrulama koşusu hazırlanır (yeni örnekler, önceden gönderilen protokol)."
                                   if decision["confirmation_run"] else
                                   "D1 ve D2 birlikte sağlanmadı: doğrulama koşusu hazırlanmaz; sonuç depoya kısa bir kayıt olarak yazılır.")]
else:
    lines.append("Koşu tamamlanınca değerlendirilir.")
flags = []
if len(per_instance):
    if (per_instance["lkh6_stopped_from_outside"] < W).any():
        flags.append("Bazı LKH süreçleri bütçeden önce kendiliğinden bitti: "
                     + ", ".join(per_instance.loc[per_instance["lkh6_stopped_from_outside"] < W, "instance"]))
    late = per_instance[per_instance["lkh6_latest_first_tour_s"] > min(SPLITS) * per_instance["budget_s"]]
    if len(late):
        flags.append("İlk bölmeden sonra ilk turunu bitiren süreç var: " + ", ".join(late["instance"]))
    if per_instance["session"].nunique() > 1 or per_instance["cpu_model"].nunique() > 1:
        flags.append(f"Koşu {per_instance['session'].nunique()} oturumda, {per_instance['cpu_model'].nunique()} işlemci modelinde yapıldı.")
lines += ["", "## Uyarılar", ""] + (["- " + x for x in flags] if flags else ["Yok."])
lines += ["", "Pilot keşif amaçlıdır: örnek sayısı küçüktür, karşılaştırmalar düzeltilmemiştir ve hiçbiri doğrulayıcı test değildir. "
          "GA-EAX tek çekirdekte koşar; birleştirme kollarında LKH durdurulduktan sonra diğer beş çekirdek boş kalır.", ""]
report = "\n".join(lines)
(OUTPUT_ROOT / "summary" / "report.md").write_text(report, encoding="utf-8")
write_json_atomic(OUTPUT_ROOT / "summary" / "main.json",
                  dict(note=note, suite_complete=bool(suite_complete), instances=len(records), total=len(INSTANCES),
                       primary_arm=PRIMARY_ARM, control_arm=CONTROL_ARM, decision=decision,
                       reference_done=int(sum(r["reference"] is not None for r in records))))
print(report)
