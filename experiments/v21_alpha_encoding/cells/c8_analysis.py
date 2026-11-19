# (anahtar, ilk kol, ikinci kol, açıklama). Yüzde fark = 100 x (ilk - ikinci) / ikinci; negatif: ilk kol daha kısa.
COMPARISONS = [
    ("P1", "pop_rank", "pop_native", "Dosyanın etkisi, POPMUSIC listesi: run v5 dosyası / LKH eşdeğeri"),
    ("P2", "beta_tree", "pop_full", "BetaTour (ağaç kodlaması) / kesintisiz POPMUSIC"),
    ("S1", "pop_native", "pop_full", "Yeniden başlatmanın kendisi: LKH eşdeğeri dosya / kesintisiz"),
    ("S2", "pop_rank", "pop_full", "Run v5'teki karşılaştırma: yeniden başlatılan POPMUSIC / kesintisiz"),
    ("S3", "beta_tree", "pop_native", "Listenin etkisi, düzeltilmiş dosyalar: BetaTour / POPMUSIC"),
    ("S4", "beta_rank", "pop_rank", "Listenin etkisi, run v5 dosyaları: BetaTour / POPMUSIC"),
    ("S5", "beta_rank", "beta_tree", "Dosyanın etkisi, BetaTour listesi: run v5 dosyası / ağaç kodlaması"),
    ("S6", "beta_tree", "freq_tree", "Model / sayım (ağaç kodlaması)"),
    ("S7", "freq_tree", "pop_full", "Sayım (ağaç kodlaması) / kesintisiz POPMUSIC"),
    ("S8", "beta_rank", "pop_full", "Run v5'teki karşılaştırma: BetaTour / kesintisiz"),
]
# Önceden belirlenen birincil uç noktalar (PROTOCOL_v21.md): bütçe sonunda, örnek düzeyinde işaret testi.
PRIMARY = {"P1": "mean6", "P2": "best6"}
ENDPOINTS = ("best6", "mean6")


def pct(a, b):
    return 100.0 * (a - b) / b


def time_tag(f):
    return f"{int(round(100 * f)):03d}"


def load_finished():
    by_name = {x["name"]: x for x in INSTANCES}
    found = []
    for path in sorted((OUTPUT_ROOT / "instances").glob("*.json")):
        item = json.loads(path.read_text())
        if item.get("complete") and item["name"] in by_name:
            assert set(item["arms"]) == set(ARM_NAMES) and item["xy_sha256"] == xy_hash(by_name[item["name"]])
            assert all(len(item["arms"][arm]["runs"]) == W for arm in ARM_NAMES)
            found.append(item)
    return sorted(found, key=lambda r: r["ordinal"])


records = load_finished()
rows, seed_rows = [], []
for item in records:
    row = dict(instance=item["name"], group=item["group"], n=item["n"], budget_s=item["budget_s"],
               ordinal=item["ordinal"], full_first=item["full_first"], control_order=" ".join(item["control_order"]),
               first_match_all=all(item["first_trial_matches_uninterrupted"]), phase1_s=item["phase1"]["wall_s"],
               model_s=item["lists"]["model_predict_s"], beta_top3_in_pop5=item["lists"]["beta_top3_in_pop5"],
               pop_candidates_per_node=item["lists"]["pop_candidates_per_node"],
               cpu_model=item["hardware"]["cpu_model"], gpu=item["hardware"]["gpu"], session=item["session_id"][:8])
    for arm in ARM_NAMES:
        data = item["arms"][arm]
        row[arm] = int(data["best_at_budget"])
        row[arm + "_mean6"] = float(data["mean_at_budget"])
        row[arm + "_trials"] = float(data["mean_trials_at_fraction"][-1])
        if arm in RESTART_ARMS:
            row[arm + "_kept"] = float(item["lists"]["kept_share"][arm])
            row[arm + "_zero_alpha"] = float(item["lists"]["zero_alpha_per_node"][arm])
            row[arm + "_prelude_s"] = float(data["prelude_s"])
            row[arm + "_overhead_s"] = float(item["lists"]["overhead_s"][arm])
        for run in data["runs"]:
            seed_rows.append(dict(instance=item["name"], group=item["group"], seed=run["seed"], arm=arm,
                                  cost=run["best_at_budget"], trials=run["trials_at_fraction"][-1]))
    for key, first, second, _ in COMPARISONS:
        for j, f in enumerate(FRACTIONS):
            a, b = item["arms"][first]["best_at_fraction"][j], item["arms"][second]["best_at_fraction"][j]
            row[f"{key}_{time_tag(f)}_best6"] = pct(a, b) if a is not None and b is not None else float("nan")
            sa = [r["best_at_fraction"][j] for r in item["arms"][first]["runs"]]
            sb = [r["best_at_fraction"][j] for r in item["arms"][second]["runs"]]
            complete = all(x is not None for x in sa + sb)
            row[f"{key}_{time_tag(f)}_mean6"] = (float(np.mean([pct(x, y) for x, y in zip(sa, sb)]))
                                                  if complete else float("nan"))
    rows.append(row)

per_instance = pd.DataFrame(rows)
per_seed = pd.DataFrame(seed_rows)
per_instance.to_csv(OUTPUT_ROOT / "summary" / "per_instance.csv", index=False)
per_seed.to_csv(OUTPUT_ROOT / "summary" / "per_seed.csv", index=False)

stat_rows, arm_rows = [], []
if len(per_instance):
    for group, frame in [("all", per_instance)] + list(per_instance.groupby("group", sort=True)):
        for ci, (key, first, second, label) in enumerate(COMPARISONS):
            for j, f in enumerate(FRACTIONS):
                for ei, endpoint in enumerate(ENDPOINTS):
                    values = frame[f"{key}_{time_tag(f)}_{endpoint}"].dropna().to_numpy()
                    if len(values):
                        stat_rows.append(dict(group=group, key=key, comparison=label, first=first, second=second,
                                              endpoint=endpoint, time_pct=int(round(100 * f)),
                                              primary_endpoint=bool(PRIMARY.get(key) == endpoint),
                                              primary=bool(group == "all" and f == FRACTIONS[-1]
                                                           and PRIMARY.get(key) == endpoint),
                                              **A.paired_stats(values, 20261119 + 1000 * ci + 10 * j + ei)))
        full_trials = frame["pop_full_trials"].mean()
        for arm in ARM_NAMES:
            restart = arm in RESTART_ARMS
            arm_rows.append(dict(group=group, arm=arm, instances=len(frame),
                                 mean_trials_at_budget=frame[arm + "_trials"].mean(),
                                 trials_vs_pop_full_pct=pct(frame[arm + "_trials"].mean(), full_trials),
                                 kept_share=frame[arm + "_kept"].mean() if restart else float("nan"),
                                 zero_alpha_per_node=frame[arm + "_zero_alpha"].mean() if restart else float("nan"),
                                 prelude_s=frame[arm + "_prelude_s"].mean() if restart else 0.0,
                                 list_overhead_s=frame[arm + "_overhead_s"].mean() if restart else 0.0))
stats = pd.DataFrame(stat_rows)
arm_table = pd.DataFrame(arm_rows)
if len(stats):
    stats["sign_p_holm"] = np.nan
    primary_index = list(stats.index[stats["primary"]])
    for j, value in zip(primary_index, A.holm([stats.loc[j, "sign_p"] for j in primary_index])):
        stats.loc[j, "sign_p_holm"] = value
stats.to_csv(OUTPUT_ROOT / "summary" / "paired_stats.csv", index=False)
arm_table.to_csv(OUTPUT_ROOT / "summary" / "arms.csv", index=False)

suite_complete = len(INSTANCES) > 0 and len(records) == len(INSTANCES)
if TEST_MODE:
    note = "TEKNİK ÖZ SINAMA: bu sonuç makale için kullanılmaz."
elif not suite_complete:
    note = f"ARA SONUÇ: {len(records)}/{len(INSTANCES)} örnek tamamlandı. p değerleri yalnızca keşif amaçlıdır; tam sonucu bekleyin."
else:
    note = f"{len(records)}/{len(INSTANCES)} örnek tamamlandı."


def verdict(row, better, worse):
    if not suite_complete or TEST_MODE:
        return "Koşu tamamlanmadan yorum yapılmaz."
    low, high = row["ci95_low_pct"], row["ci95_high_pct"]
    detail = (f"{row['win']}/{row['tie']}/{row['loss']} (ilk kol kısa/eşit/uzun), ortalama fark %{row['mean_delta_pct']:+.4f} "
              f"(%95 GA {low:+.4f} ile {high:+.4f}), işaret testi p = {row['sign_p']:.4f}, Holm p = {row['sign_p_holm']:.4f}")
    if row["sign_p_holm"] < 0.05 and row["win"] > row["loss"]:
        return better + " " + detail
    if row["sign_p_holm"] < 0.05 and row["loss"] > row["win"]:
        return worse + " " + detail
    return "Anlamlı fark saptanmadı (bu, fark olmadığının kanıtı değildir; güven aralığına bakın). " + detail


columns = ["key", "comparison", "endpoint", "instances", "win", "tie", "loss", "mean_delta_pct", "median_delta_pct",
           "ci95_low_pct", "ci95_high_pct", "sign_p"]
lines = ["# BetaTour run v21: aday dosyası kodlaması denetimi (alfa ve baba alanları)", "", note, "",
         "Yüzde fark = 100 x (ilk kol - ikinci kol) / ikinci kol; negatif değer ilk kolun daha kısa tur bulduğunu gösterir.",
         "win/tie/loss: ilk kolun daha kısa, eşit, daha uzun olduğu örnek sayısı. Güven aralığı örnek düzeyinde 5000",
         "bootstrap tekrarından, p değeri kesin iki yönlü işaret testinden gelir.",
         "best6: altı tohumun en iyisi. mean6: aynı tohumlu eşli yüzde farkların altı tohum ortalaması.",
         "Birincil karşılaştırmalar (PROTOCOL_v21.md, koşudan önce yazıldı): P1 (mean6) ve P2 (best6), bütçe sonunda,",
         "ikisi birlikte Holm düzeltmeli. Diğer bütün satırlar ikincildir ve keşif amaçlıdır.", ""]
mismatch = [r["name"] for r in records if not all(r["first_trial_matches_uninterrupted"])]
rule_mismatch = [r["name"] for r in records if any(r["lists"]["tree_rule_mismatch"])]
sessions = sorted({r["session_id"][:8] for r in records})
cpu_models = sorted({r["hardware"]["cpu_model"] for r in records})
if mismatch:
    lines += [f"UYARI: {len(mismatch)} örnekte kesintisiz kolun ilk denemesi ortak ilk aşamayla aynı maliyeti vermedi: "
              + ", ".join(mismatch), ""]
if rule_mismatch:
    lines += [f"UYARI: {len(rule_mismatch)} örnekte ağaç kuralı LKH'nin alfa = 0 girdilerini aynen üretmedi: "
              + ", ".join(rule_mismatch), ""]
if len(cpu_models) > 1:
    lines += ["UYARI: örnekler birden fazla CPU modelinde koştu: " + "; ".join(cpu_models), ""]
if len(sessions) > 1:
    lines += [f"Not: koşu {len(sessions)} Colab oturumuna bölündü; her örneğin bütün kolları tek oturumda koştu.", ""]
PRIMARY_RESULT = {}
if len(stats):
    at_budget = stats[(stats["time_pct"] == 100)]
    main = at_budget[at_budget["group"] == "all"]
    primary = main[main["primary"]]
    texts = {"P1": ("Run v1-v8'in dosyası, LKH'nin kendi dosyasının eşdeğerinden DAHA İYİ sonuç verdi (beklenenin tersi).",
                    "Run v1-v8'in dosyası sonucu kötüleştirdi."),
             "P2": ("Ağaç kodlamalı BetaTour, kesintisiz POPMUSIC'ten daha kısa tur buldu.",
                    "Ağaç kodlamalı BetaTour, kesintisiz POPMUSIC'ten daha uzun tur buldu.")}
    lines += ["## Birincil karşılaştırmalar", ""]
    for _, r in primary.iterrows():
        PRIMARY_RESULT[r["key"]] = {k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()}
        lines += [f"**{r['key']} ({r['endpoint']}): {r['comparison']}.** " + verdict(r, *texts[r["key"]]), ""]
    lines += [f"## Bütçe sonunda bütün karşılaştırmalar ({len(records)} örnek birlikte)", "", "```",
              main[columns].round(5).to_string(index=False), "```", "",
              "## Ailelere göre birincil uç noktalar", "", "```",
              at_budget[(at_budget["group"] != "all") & at_budget["primary_endpoint"]]
              [["group"] + columns].round(5).to_string(index=False), "```", "",
              "## Zaman içinde (bütçenin %25, %50, %75 ve %100'ünde; mean6)", "", "```",
              stats[(stats["group"] == "all") & (stats["endpoint"] == "mean6")
                    & stats["key"].isin(["P1", "S1", "S3", "S5"])]
              [["key", "time_pct", "win", "tie", "loss", "mean_delta_pct", "ci95_low_pct", "ci95_high_pct", "sign_p"]]
              .round(5).to_string(index=False), "```", "",
              "## Kollar: tamamlanan deneme sayısı ve yeniden başlatmada korunabilen kenar payı", "",
              "kept_share: ilk denemenin turundaki kenarlardan, aday dosyasında alfası 0 olanların payı (LKH'nin yeniden",
              "başlatma kuralının koruyabildiği kenarlar). mean_trials_at_budget: bütçe içinde tamamlanan deneme sayısı",
              "(tohum ortalaması); yeniden başlatılan kollarda ortak ilk deneme ve dosyadan okunan turu yeniden",
              "iyileştiren deneme dahildir.", "", "```",
              arm_table.round(4).to_string(index=False), "```", ""]
lines += ["Ham tablolar: summary/per_instance.csv, per_seed.csv, paired_stats.csv (bütün zaman noktaları), arms.csv.",
          "Aday dosyaları candidates/, ilk aşama dizileri stages/<örnek>/phase1.npz içindedir."]
(OUTPUT_ROOT / "summary" / "RAPOR.md").write_text("\n".join(lines), encoding="utf-8")
write_json_atomic(OUTPUT_ROOT / "summary" / "primary.json",
                  dict(complete=bool(suite_complete), instances=len(records), primary=PRIMARY_RESULT,
                       first_trial_mismatch=mismatch, tree_rule_mismatch=rule_mismatch, sessions=sessions,
                       gpus=sorted({r["hardware"]["gpu"] for r in records})))
print("\n".join(lines))
