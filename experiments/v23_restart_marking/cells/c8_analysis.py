# (anahtar, ilk kol, ikinci kol, açıklama). Yüzde fark = 100 x (ilk - ikinci) / ikinci; negatif: ilk kol daha kısa.
COMPARISONS = [
    ("M1", "beta_model", "beta_tree", "Model işaretlemesi / ağaç işaretlemesi (BetaTour listesi)"),
    ("M2", "beta_count", "beta_tree", "Sayım işaretlemesi / ağaç işaretlemesi (BetaTour listesi)"),
    ("M3", "pop_model", "pop_native", "Model işaretlemesi / ağaç işaretlemesi (POPMUSIC listesi)"),
    ("S1", "beta_model", "beta_count", "Model / sayım işaretlemesi (BetaTour listesi)"),
    ("S2", "beta_model", "pop_full", "BetaTour + model işaretlemesi / kesintisiz POPMUSIC"),
    ("S3", "beta_count", "pop_full", "BetaTour + sayım işaretlemesi / kesintisiz POPMUSIC"),
    ("S4", "beta_tree", "pop_full", "BetaTour, ağaç işaretlemesi / kesintisiz POPMUSIC (run v21'deki P2, yeni örnekler)"),
    ("S5", "pop_model", "pop_full", "POPMUSIC + model işaretlemesi / kesintisiz POPMUSIC"),
    ("S6", "pop_native", "pop_full", "Yeniden başlatmanın kendisi: LKH eşdeğeri dosya / kesintisiz"),
    ("S7", "beta_tree", "pop_native", "Listenin etkisi, ağaç işaretlemesi: BetaTour / POPMUSIC"),
    ("S8", "beta_model", "pop_model", "Listenin etkisi, model işaretlemesi: BetaTour / POPMUSIC"),
]
# Önceden belirlenen üç ana okuma (PROTOCOL_v23.md): bütçe sonunda, altı tohumun eşli ortalaması. Koşu keşif
# amaçlıdır; hiçbir satır doğrulayıcı bir test değildir.
MAIN = {"M1": "mean6", "M2": "mean6", "M3": "mean6"}
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
               tour_edges_in_pop=item["lists"]["tour_edges_in_list"]["pop"],
               tour_edges_in_beta=item["lists"]["tour_edges_in_list"]["beta"],
               model_threshold=float(np.mean(item["lists"]["model_threshold"])),
               first_tour_trusted_by_count=float(np.mean(item["lists"]["first_tour_edges_trusted_by_count"])),
               first_tour_trusted_by_model=float(np.mean(item["lists"]["first_tour_edges_trusted_by_model"])),
               first_tour_trusted_by_both=float(np.mean(item["lists"]["first_tour_edges_trusted_by_both"])),
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
                                              main_endpoint=bool(MAIN.get(key) == endpoint),
                                              main=bool(group == "all" and f == FRACTIONS[-1]
                                                        and MAIN.get(key) == endpoint),
                                              **A.paired_stats(values, 20261123 + 1000 * ci + 10 * j + ei)))
        full_trials = frame["pop_full_trials"].mean()
        # En kısa tur: bir örnekte birden fazla kol eşitse her biri sayılır.
        best_cost = frame[ARM_NAMES].min(axis=1)
        best_mean = frame[[arm + "_mean6" for arm in ARM_NAMES]].min(axis=1)
        for arm in ARM_NAMES:
            restart = arm in RESTART_ARMS
            arm_rows.append(dict(group=group, arm=arm, instances=len(frame),
                                 shortest_best6=int((frame[arm] == best_cost).sum()),
                                 shortest_mean6=int((frame[arm + "_mean6"] == best_mean).sum()),
                                 mean_trials_at_budget=frame[arm + "_trials"].mean(),
                                 trials_vs_pop_full_pct=pct(frame[arm + "_trials"].mean(), full_trials),
                                 kept_share=frame[arm + "_kept"].mean() if restart else float("nan"),
                                 zero_alpha_per_node=frame[arm + "_zero_alpha"].mean() if restart else float("nan"),
                                 prelude_s=frame[arm + "_prelude_s"].mean() if restart else 0.0,
                                 list_overhead_s=frame[arm + "_overhead_s"].mean() if restart else 0.0))
stats = pd.DataFrame(stat_rows)
arm_table = pd.DataFrame(arm_rows)
stats.to_csv(OUTPUT_ROOT / "summary" / "paired_stats.csv", index=False)
arm_table.to_csv(OUTPUT_ROOT / "summary" / "arms.csv", index=False)

suite_complete = len(INSTANCES) > 0 and len(records) == len(INSTANCES)
if TEST_MODE:
    note = "TEKNİK ÖZ SINAMA: bu sonuç makale için kullanılmaz."
elif not suite_complete:
    note = f"ARA SONUÇ: {len(records)}/{len(INSTANCES)} örnek tamamlandı. Tam sonucu bekleyin."
else:
    note = f"{len(records)}/{len(INSTANCES)} örnek tamamlandı."


def reading(row):
    """Önceden yazılan karar kuralı: örnek düzeyinde kesin iki yönlü işaret testi p < 0,05 ise (12 örnekte eşitlik
    yoksa en az 10 örnek) ve yeni işaretleme daha çok örnekte kısaysa doğrulama koşusu hazırlanır."""
    if not suite_complete or TEST_MODE:
        return "undecided", "Koşu tamamlanmadan yorum yapılmaz."
    low, high = row["ci95_low_pct"], row["ci95_high_pct"]
    detail = (f"{row['win']}/{row['tie']}/{row['loss']} (ilk kol kısa/eşit/uzun), işaret testi p = {row['sign_p']:.4f} "
              f"(düzeltmesiz, keşif amaçlı), ortalama fark %{row['mean_delta_pct']:+.4f} (%95 GA {low:+.4f} ile {high:+.4f}), "
              f"ortanca %{row['median_delta_pct']:+.4f}")
    if row["sign_p"] < 0.05 and row["win"] > row["loss"]:
        return "confirm", "Yeni işaretleme daha kısa tur verdi: DOĞRULAMA KOŞUSU hazırlanır. " + detail
    if row["sign_p"] < 0.05 and row["loss"] > row["win"]:
        return "worse", "Yeni işaretleme daha uzun tur verdi: fikir bu biçimiyle bırakılır. " + detail
    return "no_difference", ("Karar kuralı sağlanmadı: doğrulama koşusu hazırlanmaz (bu, fark olmadığının kanıtı "
                             "değildir; güven aralığına bakın). " + detail)


columns = ["key", "comparison", "endpoint", "instances", "win", "tie", "loss", "mean_delta_pct", "median_delta_pct",
           "ci95_low_pct", "ci95_high_pct", "sign_p"]
lines = ["# BetaTour run v23: yeniden başlatma işaretlemesi (keşif koşusu)", "", note, "",
         "Bu koşu keşif amaçlıdır: p değerleri düzeltmesizdir ve hiçbir satır doğrulayıcı bir test değildir. Daha iyi",
         "görünen bir işaretleme, makalede yer almadan önce kendi protokolü olan ayrı bir koşuda sınanır.", "",
         "Yüzde fark = 100 x (ilk kol - ikinci kol) / ikinci kol; negatif değer ilk kolun daha kısa tur bulduğunu gösterir.",
         "win/tie/loss: ilk kolun daha kısa, eşit, daha uzun olduğu örnek sayısı. Güven aralığı örnek düzeyinde 5000",
         "bootstrap tekrarından, p değeri kesin iki yönlü işaret testinden gelir.",
         "best6: altı tohumun en iyisi. mean6: aynı tohumlu eşli yüzde farkların altı tohum ortalaması.",
         "Ana okumalar (PROTOCOL_v23.md, koşudan önce yazıldı): M1, M2 ve M3; mean6, bütçe sonunda. Karar kuralı:",
         "işaret testi p < 0,05 ve yeni işaretleme daha çok örnekte kısaysa o işaretleme için doğrulama koşusu hazırlanır.", ""]
mismatch = [r["name"] for r in records if not all(r["first_trial_matches_uninterrupted"])]
rule_mismatch = [r["name"] for r in records if any(r["lists"]["tree_rule_mismatch"])]
writer_mismatch = [r["name"] for r in records if any(r["lists"]["mark_writer_mismatch"])]
sessions = sorted({r["session_id"][:8] for r in records})
cpu_models = sorted({r["hardware"]["cpu_model"] for r in records})
if mismatch:
    lines += [f"UYARI: {len(mismatch)} örnekte kesintisiz kolun ilk denemesi ortak ilk aşamayla aynı maliyeti vermedi: "
              + ", ".join(mismatch), ""]
if rule_mismatch:
    lines += [f"UYARI: {len(rule_mismatch)} örnekte ağaç kuralı LKH'nin alfa = 0 girdilerini aynen üretmedi: "
              + ", ".join(rule_mismatch), ""]
if writer_mismatch:
    lines += [f"UYARI: {len(writer_mismatch)} örnekte işaretleme yazıcısı LKH'nin kendi alfa değerlerini aynen üretmedi: "
              + ", ".join(writer_mismatch), ""]
if len(cpu_models) > 1:
    lines += ["UYARI: örnekler birden fazla CPU modelinde koştu: " + "; ".join(cpu_models), ""]
if len(sessions) > 1:
    lines += [f"Not: koşu {len(sessions)} Colab oturumuna bölündü; her örneğin bütün kolları tek oturumda koştu.", ""]
MAIN_RESULT = {}
if len(stats):
    at_budget = stats[(stats["time_pct"] == 100)]
    whole = at_budget[at_budget["group"] == "all"]
    lines += ["## Ana okumalar", ""]
    for _, r in whole[whole["main"]].iterrows():
        decision, text = reading(r)
        MAIN_RESULT[r["key"]] = dict({k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()},
                                     decision=decision)
        lines += [f"**{r['key']} ({r['endpoint']}): {r['comparison']}.** " + text, ""]
    lines += [f"## Bütçe sonunda bütün karşılaştırmalar ({len(records)} örnek birlikte)", "", "```",
              whole[columns].round(5).to_string(index=False), "```", "",
              "## Ailelere göre ana okumalar", "", "```",
              at_budget[(at_budget["group"] != "all") & at_budget["main_endpoint"]]
              [["group"] + columns].round(5).to_string(index=False), "```", "",
              "## Zaman içinde (bütçenin %25, %50, %75 ve %100'ünde; mean6)", "", "```",
              stats[(stats["group"] == "all") & (stats["endpoint"] == "mean6") & stats["key"].isin(list(MAIN))]
              [["key", "time_pct", "win", "tie", "loss", "mean_delta_pct", "ci95_low_pct", "ci95_high_pct", "sign_p"]]
              .round(5).to_string(index=False), "```", "",
              "## Kollar: en kısa tur, deneme sayısı ve yeniden başlatmada korunabilen kenar payı", "",
              "shortest_best6 / shortest_mean6: kolun, altı kol içinde en kısa turu (altı tohumun en iyisi / ortalaması)",
              "bulduğu örnek sayısı; eşitlikte her kol sayılır. kept_share: ilk denemenin turundaki kenarlardan, aday",
              "dosyasında alfası 0 olanların payı (LKH'nin yeniden başlatma kuralının koruyabildiği kenarlar); 1 - kept_share",
              "her denemede bozulmaya açık bırakılan paydır. mean_trials_at_budget: bütçe içinde tamamlanan deneme sayısı",
              "(tohum ortalaması). list_overhead_s: kolun bütçesinden düşülen hazırlık süresi (model, liste, işaretleme).",
              "", "```", arm_table.round(4).to_string(index=False), "```", ""]
lines += ["Ham tablolar: summary/per_instance.csv, per_seed.csv, paired_stats.csv (bütün zaman noktaları), arms.csv.",
          "Aday dosyaları candidates/, ilk aşama dizileri stages/<örnek>/phase1.npz içindedir."]
(OUTPUT_ROOT / "summary" / "RAPOR.md").write_text("\n".join(lines), encoding="utf-8")
write_json_atomic(OUTPUT_ROOT / "summary" / "main.json",
                  dict(complete=bool(suite_complete), instances=len(records), main=MAIN_RESULT,
                       first_trial_mismatch=mismatch, tree_rule_mismatch=rule_mismatch,
                       mark_writer_mismatch=writer_mismatch, sessions=sessions,
                       gpus=sorted({r["hardware"]["gpu"] for r in records})))
print("\n".join(lines))
