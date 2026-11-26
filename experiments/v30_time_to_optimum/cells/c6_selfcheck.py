# Öz sınama (1-2 dk), ilk örnekten önce. Bu makinede şunları denetler:
#  1) pop kolunun tek denemelik koşusu, ilk deneme işleviyle aynı turu verir (aynı tohum);
#  2) OPTIMUM verilince LKH-3 o uzunlukta durur; ulaşılamaz OPTIMUM'da TOTAL_TIME_LIMIT'te kendisi durur ve turunu
#     yazar (işlemci saati);
#  3) BetaTour listeleriyle yeniden başlatma çalışır; 4) run v21'in kodlama öz sınaması (LKH-3'ün kendi dosyası ile
#  yeniden yazılmış dosya aynı aramayı verir; ağaç kuralı LKH-3'ün alfa = 0 girdilerini verir); 5) istatistik araçları.
SELF = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "n": SELF_N}
xy_s = T.to_int_coords(T.gen_instance(SELF_N, "uniform", np.random.default_rng(424242)))
(WORK_ROOT / "problems").mkdir(exist_ok=True)
pf_s = str(WORK_ROOT / "problems" / "selfcheck.tsp")
T.write_tsp(pf_s, xy_s, name="selfcheck")
UNREACHABLE = 1                                      # hiçbir tur bu uzunlukta değildir

first_s = V.first_trial(LKH, TMP, SEEDS[0], xy_s, pf_s, "EUC_2D", UNREACHABLE, K_LIST)
one = V.lkh_run(LKH, TMP, pf_s, xy_s, "EUC_2D", SEEDS[0], UNREACHABLE, 60.0,
                params=dict(V.UNINTERRUPTED["pop"], MAX_CANDIDATES=K_LIST), max_trials=1, return_tour=True)
SELF["first_trial_equals_pop_first_trial"] = bool(one["tour"] is not None
                                                  and np.array_equal(one["tour"], first_s["tour"].astype(np.int64)))

probe = [V.lkh_run(LKH, TMP, pf_s, xy_s, "EUC_2D", s, UNREACHABLE, 2.0, params=dict(V.UNINTERRUPTED["pop"]))
         for s in (11, 12)]
best = min(probe, key=lambda r: r["cost"])
best_seed = (11, 12)[probe.index(best)]
stop = V.lkh_run(LKH, TMP, pf_s, xy_s, "EUC_2D", best_seed, int(best["cost"]), 2.0,
                 params=dict(V.UNINTERRUPTED["pop"]))
SELF["stop_at_optimum"] = dict(target=best["cost"], found=stop["found"], time_total=stop["time_total"],
                               probe_time_total=best["time_total"])
SELF["time_limit"] = dict(limit=2.0, time_totals=[r["time_total"] for r in probe],
                          returncodes=[r["returncode"] for r in probe])

firsts_s = [V.first_trial(LKH, TMP, s, xy_s, pf_s, "EUC_2D", UNREACHABLE, K_LIST) for s in SEEDS]
succ_s = np.stack([T.tour_succ_pred(f["tour"])[0] for f in firsts_s], axis=1).astype(np.int32)
pred_s = M.predict_tourfold(MODEL, xy_s, K_GRAPH, DEVICE, ens_succ=succ_s,
                            ens_cost=np.array([f["cost"] for f in firsts_s]))       # GPU ısınması
pred_s = M.predict_tourfold(MODEL, xy_s, K_GRAPH, DEVICE, ens_succ=succ_s,
                            ens_cost=np.array([f["cost"] for f in firsts_s]))
lists_s, build_s, _, _ = V.arm_lists(firsts_s, pred_s, k=K_LIST, k_union=K_UNION, k_learned=K_LEARNED)
restart = {}
for arm in V.RESTARTED:
    r = V.lkh_run(LKH, TMP, pf_s, xy_s, "EUC_2D", SEEDS[0], UNREACHABLE, 1.0, files=lists_s[arm][0],
                  init_tour=firsts_s[0]["tour"])
    restart[arm] = dict(cost=r["cost"], returncode=r["returncode"], time_total=r["time_total"],
                        not_longer_than_first=bool(r["cost"] is not None and r["cost"] <= firsts_s[0]["cost"]))
SELF["restart"] = restart
SELF["model_s"] = float(pred_s["t_total"])
SELF["list_build_s"] = build_s
SELF["encoding"] = A.encoding_selfcheck(LKH, TMP, xy_s, SEEDS[0], trials=25)
SELF["statistics"] = dict(holm=V.holm([0.01, 0.04]).tolist(), compare=V.compare([1, 2, 3, 5], [2, 2, 4, 4]),
                          par=V.par([1.0, float("inf")], 10.0))
write_json_atomic(OUTPUT_ROOT / "selfcheck" / "selfcheck.json", SELF)

checks = {
    "tek denemelik pop koşusu = ilk deneme": SELF["first_trial_equals_pop_first_trial"],
    "OPTIMUM'da durur": stop["found"] and stop["time_total"] is not None and stop["time_total"] <= best["time_total"] + 0.05,
    "işlemci saati sınırında kendisi durur": all(r["returncode"] == 0 and r["time_total"] is not None
                                                 and 1.9 <= r["time_total"] <= 2.6 for r in probe),
    "yeniden başlatmalar çalışır": all(v["returncode"] == 0 and v["not_longer_than_first"] for v in restart.values()),
    "kendi dosyası = LKH dosyası (run v21)": SELF["encoding"]["native_file_equals_lkh_file"],
    "ağaç kuralı (run v21)": SELF["encoding"]["tree_rule_matches_lkh"],
    "Holm ve işaret testi": np.allclose(SELF["statistics"]["holm"], [0.02, 0.04])
    and SELF["statistics"]["compare"]["wins"] == 2 and SELF["statistics"]["compare"]["losses"] == 1
    and abs(SELF["statistics"]["par"] - 50.5) < 1e-9,
}
SELF["checks"] = {k: bool(v) for k, v in checks.items()}
write_json_atomic(OUTPUT_ROOT / "selfcheck" / "selfcheck.json", SELF)
for k, v in checks.items():
    print(("  tamam  " if v else "  HATA   ") + k)
print(f"BetaTour çıkarımı {SELF['model_s']:.2f} s ({SELF_N} düğüm); OPTIMUM denemesi: hedef {best['cost']:.0f}, "
      f"durduğu süre {stop['time_total']} s (deneme koşusunda {best['time_total']} s).")
assert all(checks.values()), "Öz sınama geçmedi; selfcheck/selfcheck.json dosyasını gönderin."
