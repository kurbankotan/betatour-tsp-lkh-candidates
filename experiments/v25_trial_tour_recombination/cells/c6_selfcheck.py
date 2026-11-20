# Öz sınama (ölçüme dahil değildir). Bu makinede ve derlenen ikililerle:
#   1) Kayıt eden LKH, değiştirilmemiş LKH ile aynı aramayı yapıyor mu? Sabit deneme sayılı (belirlenimci) iki koşu
#      her denemede aynı maliyeti vermelidir.
#   2) Kaydedilen her tur geçerli mi ve uzunluğu kaydedilen maliyete eşit mi? Son "en iyi tur" kaydı, LKH'nin
#      kendi yazdığı tur mu?
#   3) GA-EAX kaydedilen turlardan başlayınca geçerli bir tur veriyor mu, bildirdiği uzunluk kendi hesabımızla aynı
#      mı, en iyi girdi turundan uzun olmayan bir tur mu? İki sürüm de (Normal, Large) denenir.
#   4) GA-EAX duvar saati sınırında en iyi turuyla duruyor mu?
#   5) En yakın şehir listelerini ızgarayla bulan GA-EAX, bu yama olmadan derlenen GA-EAX ile aynı turu, aynı kuşak
#      sayısında veriyor mu?
def selfcheck(n=2000, trials=25):
    xy = T.to_int_coords(T.gen_instance(n, "clustered", np.random.default_rng(9093)))
    folder = WORK_ROOT / "selfcheck"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    problem, eax_problem = folder / "p.tsp", folder / "p_eax.tsp"
    T.write_tsp(str(problem), xy, name="selfcheck", ew="EUC_2D")
    B.write_eax_problem(str(eax_problem), xy)

    def lkh(binary, tag, dump):
        par = folder / (tag + ".par")
        B.write_par(par, dict(PROBLEM_FILE=problem, RUNS=1, TRACE_LEVEL=2, MAX_TRIALS=trials, SEED=SEEDS[0],
                              **POP_PARAMS, OUTPUT_TOUR_FILE=folder / (tag + ".tour")))
        res = B.run_process([binary, par], folder, env={"BETATOUR_TRIAL_DUMP": dump} if dump else None, timeout=900)
        assert res["returncode"] == 0, res["stdout"][-500:]
        return res

    plain, recorded = lkh(LKH, "plain", None), lkh(LKH_REC, "rec", folder / "rec.bin")
    costs_plain, costs_rec = B.lkh_trial_costs(plain["stdout"]), B.lkh_trial_costs(recorded["stdout"])
    records = B.read_records(folder / "rec.bin", n)
    trial_rec, best_rec = records[records["kind"] == 0], records[records["kind"] == 1]
    valid = all(B.is_permutation(B.tour_of(r), n) and int(T.tour_length(xy, B.tour_of(r))) == int(r["cost"])
                for r in records)
    own = T.read_tour(str(folder / "rec.tour"))
    tours, costs, _ = B.select_population([records], int(records["epoch_ms"][-1]), 12, n)
    B.write_population(folder / "pop", tours, costs)
    eax = {}
    for version in ("Normal", "Large"):
        out, best_file = B.run_eax(EAX[version], str(eax_problem), str(folder / ("eax_" + version)), "S", len(tours),
                                   kids=EAX_KIDS, population_file=str(folder / "pop"), deadline_epoch=time.time() + 120)
        tour, value = B.read_eax_best(best_file, n)
        ref, ref_file = B.run_eax(EAX_PLAIN[version], str(eax_problem), str(folder / ("plain_" + version)), "S",
                                  len(tours), kids=EAX_KIDS, population_file=str(folder / "pop"),
                                  deadline_epoch=time.time() + 120)
        ref_tour, ref_value = B.read_eax_best(ref_file, n)
        eax[version] = dict(finished=bool(out["finished"]), valid=bool(B.is_permutation(tour, n)),
                            length_reported=int(value), length_recomputed=int(T.tour_length(xy, tour)),
                            generations=out["generations"], wall_s=float(out["wall_s"]),
                            setup_s=out["first_line_s"], setup_without_grid_s=ref["first_line_s"],
                            same_as_without_grid=bool(ref_value == value and ref["generations"] == out["generations"]
                                                      and np.array_equal(ref_tour, tour)))
    # Sınır şimdiki an: GA-EAX popülasyonunu kurar, ilk denetimde durur ve o andaki en iyi turunu yazar.
    out, best_file = B.run_eax(EAX["Normal"], str(eax_problem), str(folder / "eax_deadline"), "D", EAX_SCRATCH_POP,
                               kids=EAX_KIDS, deadline_epoch=time.time())
    tour, value = B.read_eax_best(best_file, n)
    result = dict(
        nodes=n, trials=trials,
        same_search=bool(len(costs_plain) == trials and costs_plain == costs_rec),
        trial_tours_recorded=int(len(trial_rec)), best_tours_recorded=int(len(best_rec)),
        recorded_tours_valid=bool(valid and list(trial_rec["trial"]) == list(range(1, trials + 1))),
        last_best_is_lkh_tour=bool(np.array_equal(B.edge_keys(own, n), B.edge_keys(B.tour_of(best_rec[-1]), n))
                                   and int(best_rec[-1]["cost"]) == min(c for _, c in costs_rec)),
        wall_plain_s=float(plain["wall"]), wall_recording_s=float(recorded["wall"]),
        best_input=int(min(costs)), eax=eax,
        deadline=dict(finished=bool(out["finished"]), stopped_by_deadline=bool(out["stopped_by_deadline"]),
                      valid=bool(B.is_permutation(tour, n) and int(T.tour_length(xy, tour)) == int(value)),
                      wall_s=float(out["wall_s"])))
    shutil.rmtree(folder, ignore_errors=True)
    return result


SELFCHECK = selfcheck(n=600, trials=12) if TEST_MODE else selfcheck()
SELFCHECK["session_id"] = SESSION_ID
write_json_atomic(OUTPUT_ROOT / "summary" / "selfcheck.json", SELFCHECK)
assert SELFCHECK["same_search"], "Kayıt eden LKH, değiştirilmemiş LKH ile aynı aramayı yapmadı."
assert SELFCHECK["recorded_tours_valid"] and SELFCHECK["last_best_is_lkh_tour"], "Kaydedilen turlar beklenen gibi değil."
for version, item in SELFCHECK["eax"].items():
    assert item["finished"] and item["valid"] and item["length_reported"] == item["length_recomputed"], \
        f"GA-EAX ({version}) geçerli ve uzunluğu doğru bir tur vermedi."
    assert item["length_reported"] <= SELFCHECK["best_input"], f"GA-EAX ({version}) en iyi girdi turundan uzun tur verdi."
    assert item["same_as_without_grid"], f"GA-EAX ({version}): ızgaralı komşu araması yamasız derlemeyle aynı sonucu vermedi."
assert SELFCHECK["deadline"]["finished"] and SELFCHECK["deadline"]["stopped_by_deadline"] and SELFCHECK["deadline"]["valid"], \
    "GA-EAX duvar saati sınırında en iyi turuyla durmadı."
print("Öz sınama tamam. Kayıt eden LKH aynı aramayı yaptı (" + str(SELFCHECK["trials"]) + " deneme; süre "
      f"{SELFCHECK['wall_plain_s']:.1f} s / kayıtla {SELFCHECK['wall_recording_s']:.1f} s). GA-EAX: "
      + ", ".join(f"{v} {x['length_reported']} ({x['generations']} kuşak)" for v, x in SELFCHECK["eax"].items())
      + f"; en iyi girdi turu {SELFCHECK['best_input']}.")
