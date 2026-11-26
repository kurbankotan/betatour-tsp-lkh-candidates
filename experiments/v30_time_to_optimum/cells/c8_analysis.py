# Analiz ve karar (saniyeler). Birincil: H1 (beta < pop) ve H2 (beta < alpha), PAR10 üzerinde örnek bazında kesin
# işaret testi, Holm. Geri kalanı betimseldir. Tablolar ekrana yazılır ve summary/ klasörüne kaydedilir.
INF = float("inf")
records = [read_json(run_path(inst, arm, j)) for inst in INSTANCES for arm in ARMS for j in range(len(SEEDS))]
MISSING = sum(r is None for r in records)
RUNS = pd.DataFrame([r for r in records if r is not None])
RUNS["success"] = RUNS["tto"].notna()

rows = []
for inst in INSTANCES:
    for arm in ARMS:
        sub = RUNS[(RUNS["instance"] == inst["name"]) & (RUNS["arm"] == arm)]
        times = [INF if t is None or (isinstance(t, float) and math.isnan(t)) else float(t) for t in sub["tto"]]
        rows.append(dict(instance=inst["name"], group=inst["group"], n=inst["n"], arm=arm, cap=inst["cap"],
                         runs=len(sub), successes=int(sub["success"].sum()),
                         par10=V.par(times, inst["cap"], PAR_FACTOR) if times else np.nan,
                         par2=V.par(times, inst["cap"], 2.0) if times else np.nan,
                         median=float(np.median(times)) if times else np.nan,
                         pause=float(sub["pause"].mean()) if len(sub) else np.nan))
PER = pd.DataFrame(rows)
COMPLETE = [inst["name"] for inst in INSTANCES
            if (PER[PER["instance"] == inst["name"]]["runs"] == len(SEEDS)).all()]
WIDE = PER.pivot(index="instance", columns="arm", values="par10").loc[COMPLETE, ARMS]
GROUP_OF = {inst["name"]: inst["group"] for inst in INSTANCES}


CAP_OF = {inst["name"]: inst["cap"] for inst in INSTANCES}


def paired(first, second, names=None, seed=BOOTSTRAP_SEED):
    names = list(WIDE.index) if names is None else names
    tol = np.array([TIE_FRACTION * CAP_OF[name] for name in names])
    out = V.compare(WIDE.loc[names, first], WIDE.loc[names, second], tol=tol)
    out.update(V.ratio_interval(WIDE.loc[names, first], WIDE.loc[names, second], seed))
    out["instances"] = len(names)
    return out


h1, h2 = paired("beta", "pop"), paired("beta", "alpha")
adjusted = V.holm([h1["p"], h2["p"]])
h1["p_holm"], h2["p_holm"] = float(adjusted[0]), float(adjusted[1])
H1 = bool(h1["wins"] > h1["losses"] and h1["p_holm"] < 0.05)
H2 = bool(h2["wins"] > h2["losses"] and h2["p_holm"] < 0.05)
SECONDARY = paired("beta", "count")
CREDITED = bool(SECONDARY["wins"] > SECONDARY["losses"] and SECONDARY["p"] < 0.05)
if H1 and H2:
    VERDICT = ("H1 ve H2 desteklendi: BetaTour listeleriyle LKH-3 kanıtlı optimuma kendi listelerinden daha çabuk "
               "ulaşıyor. " + ("Öğrenilmiş sıralamaya pay verilir (beta, sayma listesini de yeniyor)."
                               if CREDITED else "Kazanç, ilk deneme turlarından kurulan listelere verilir (beta, sayma "
                                                "listesinden belirgin biçimde iyi değil)."))
elif H1 or H2:
    VERDICT = f"Kısmi sonuç: yalnızca {'H1 (pop)' if H1 else 'H2 (alpha)'} desteklendi."
else:
    VERDICT = "H1 ve H2 desteklenmedi: optimuma ulaşma süresinde üstünlük yok (olumsuz sonuç olarak raporlanır)."

PAIRS = [("beta", "pop"), ("beta", "alpha"), ("beta", "count"), ("beta", "pop_r"), ("pop_r", "pop"),
         ("mix", "beta"), ("union", "beta"), ("count", "pop"), ("count", "alpha"), ("mix", "pop"), ("union", "pop"),
         ("alpha", "pop")]
groups = list(dict.fromkeys(GROUP_OF[name] for name in WIDE.index))
tsplib_names = [name for name in WIDE.index if GROUP_OF[name] == "tsplib"]
synthetic_names = [name for name in WIDE.index if GROUP_OF[name] != "tsplib"]
pair_rows = []
for a, b in PAIRS:
    for label, names in [("hepsi", list(WIDE.index)), ("sentetik", synthetic_names), ("TSPLIB", tsplib_names)]:
        if len(names) >= 2:
            r = paired(a, b, names)
            pair_rows.append({"karşılaştırma": f"{a} - {b}", "örnekler": label, "n": r["instances"],
                              "kazanç/eşit/kayıp": f"{r['wins']}/{r['ties']}/{r['losses']}", "p (düzeltmesiz)": r["p"],
                              "PAR10 oranı": r["ratio"], "%95 aralık": f"{r['low']:.3f}-{r['high']:.3f}"})
PAIR_TABLE = pd.DataFrame(pair_rows)

success = PER[PER["instance"].isin(WIDE.index)].groupby(["group", "arm"])["successes"].sum().unstack()[ARMS]
runs_per_group = PER[PER["instance"].isin(WIDE.index)].groupby(["group", "arm"])["runs"].sum().unstack()[ARMS]
SUCCESS_TABLE = (100.0 * success / runs_per_group).loc[groups]
GEO_TABLE = PER[PER["instance"].isin(WIDE.index)].groupby(["group", "arm"])["par10"].apply(
    lambda s: float(np.exp(np.log(s).mean()))).unstack()[ARMS].loc[groups]
PAUSE_TABLE = PER[PER["arm"].isin(V.RESTARTED)].groupby(["group", "arm"])["pause"].mean().unstack()[list(V.RESTARTED)]

coverage_rows = []
for inst in INSTANCES:
    meta = read_json(inst["dir"] / "lists.json") or {}
    missed = meta.get("missed_optimal_edges")
    if not missed:
        continue
    for arm in ARMS:
        for j, seed in enumerate(SEEDS):
            row = RUNS[(RUNS["instance"] == inst["name"]) & (RUNS["arm"] == arm) & (RUNS["seed"] == seed)]
            if len(row):
                coverage_rows.append(dict(instance=inst["name"], arm=arm, seed=seed, missed=int(missed[arm][j]),
                                          success=bool(row["success"].iloc[0])))
COVERAGE = pd.DataFrame(coverage_rows)
if len(COVERAGE):
    COVERAGE_TABLE = COVERAGE.assign(whole_tour=COVERAGE["missed"] == 0).groupby(["arm", "whole_tour"]).agg(
        koşu=("success", "size"), başarı_yüzdesi=("success", lambda s: 100.0 * s.mean())).unstack()
    MISSED_TABLE = COVERAGE.groupby("arm")["missed"].agg(["mean", "max"]).loc[ARMS]
else:
    COVERAGE_TABLE = MISSED_TABLE = pd.DataFrame()

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
print(f"Kayıtlı koşu: {len(RUNS)}/{len(records)} (eksik {MISSING}); tam örnek: {len(WIDE)}/{len(INSTANCES)}")
print("\nTablo 1. Optimuma ulaşan koşuların yüzdesi (gruplara göre)")
print(SUCCESS_TABLE.round(1).to_string())
print("\nTablo 2. PAR10 (saniye; örnekler üzerinde geometrik ortalama)")
print(GEO_TABLE.round(2).to_string())
print("\nTablo 3. Duraklama (saniye; yeniden başlatılan kollar)")
print(PAUSE_TABLE.round(2).to_string())
print("\nTablo 4. Eşli karşılaştırmalar (PAR10, örnek bazında; oran < 1: ilk kol daha çabuk)")
print(PAIR_TABLE.round(4).to_string(index=False))
if len(COVERAGE):
    print("\nTablo 5. Pilotun optimal turları olan örneklerde: listelerin kaçırdığı optimal kenar (koşu başına) ve "
          "optimal turun tamamı listelerdeyken / değilken başarı")
    print(MISSED_TABLE.round(2).to_string())
    print(COVERAGE_TABLE.round(1).to_string())
print("\nKarar kuralı (protokol)")
print(f"  H1 beta < pop  : {h1['wins']}/{h1['ties']}/{h1['losses']}, p = {h1['p']:.4g}, Holm p = {h1['p_holm']:.4g}, "
      f"PAR10 oranı {h1['ratio']:.3f} ({h1['low']:.3f}-{h1['high']:.3f}) -> {'DESTEKLENDİ' if H1 else 'desteklenmedi'}")
print(f"  H2 beta < alpha: {h2['wins']}/{h2['ties']}/{h2['losses']}, p = {h2['p']:.4g}, Holm p = {h2['p_holm']:.4g}, "
      f"PAR10 oranı {h2['ratio']:.3f} ({h2['low']:.3f}-{h2['high']:.3f}) -> {'DESTEKLENDİ' if H2 else 'desteklenmedi'}")
print(f"  İkincil beta < count: {SECONDARY['wins']}/{SECONDARY['ties']}/{SECONDARY['losses']}, "
      f"p = {SECONDARY['p']:.4g} (düzeltmesiz)")
print("  Sonuç:", VERDICT)

out_dir = OUTPUT_ROOT / "summary"
RUNS.drop(columns=[c for c in ["stdout_tail"] if c in RUNS.columns]).to_csv(out_dir / "runs.csv", index=False)
PER.to_csv(out_dir / "per_instance_arm.csv", index=False)
WIDE.to_csv(out_dir / "par10_wide.csv")
PAIR_TABLE.to_csv(out_dir / "pairs.csv", index=False)
if len(COVERAGE):
    COVERAGE.to_csv(out_dir / "coverage_runs.csv", index=False)
write_json_atomic(out_dir / "summary.json", {
    "decision": {"H1": H1, "H2": H2, "h1": h1, "h2": h2, "secondary_beta_vs_count": SECONDARY,
                 "learned_ranking_credited": CREDITED, "verdict": VERDICT},
    "complete_instances": len(WIDE), "instances": len(INSTANCES), "missing_runs": MISSING,
    "success_pct": SUCCESS_TABLE.reset_index().to_dict(orient="records"),
    "par10_geomean": GEO_TABLE.reset_index().to_dict(orient="records"),
    "pause_s": PAUSE_TABLE.reset_index().to_dict(orient="records"),
    "pairs": PAIR_TABLE.to_dict(orient="records"),
    "missed_optimal_edges": MISSED_TABLE.reset_index().to_dict(orient="records") if len(COVERAGE) else None,
    "selfcheck": SELF, "manifest": MANIFEST})
print(f"\nKaydedildi: {out_dir}")
