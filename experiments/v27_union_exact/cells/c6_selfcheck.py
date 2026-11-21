# Öz sınama (ölçüme dahil değildir). Bu makinede ve derlenen ikililerle:
#   1) Kayıt eden LKH (aday dosyasını da yazarken), değiştirilmemiş LKH ile aynı aramayı yapıyor mu? Sabit deneme
#      sayılı (belirlenimci) iki koşu her denemede aynı maliyeti vermelidir.
#   2) Kaydedilen her tur geçerli mi ve uzunluğu kaydedilen maliyete eşit mi? Son "en iyi tur" kaydı, LKH'nin
#      kendi yazdığı tur mu?
#   3) GA-EAX kaydedilen turlardan başlayınca geçerli bir tur veriyor mu, bildirdiği uzunluk kendi hesabımızla aynı
#      mı, en iyi girdi turundan uzun olmayan bir tur mu? İki sürüm de (Normal, Large) denenir.
#   4) GA-EAX duvar saati sınırında en iyi turuyla duruyor mu?
#   5) En yakın şehir listelerini ızgarayla bulan GA-EAX, bu yama olmadan derlenen GA-EAX ile aynı turu, aynı kuşak
#      sayısında veriyor mu?
#   6) LKH'nin kendi tur birleştirmesi (kontrol kolu) beklendiği gibi çalışıyor mu? Aday dosyası okunuyor mu (LKH
#      okuyamadığı dosyayı yeniden yazar), yalnızca en iyi turları kaydeden kip aramayı değiştirmiyor mu, sonuç turu
#      geçerli mi, başlangıç turundan uzun değil mi ve birleştirilen turların ortak kenarlarının hepsini taşıyor mu?
#   7) Birleşim grafında tam çözüm: küçük rastgele graflarda sayımla aynı en kısa turu mu veriyor; kayıtlı altı
#      deneme turunun birleşiminde kanıtlanmış en kısa, geçerli, yalnızca birleşimin kenarlarını kullanan ve en iyi
#      girdi turundan uzun olmayan bir tur mu veriyor; sabitlenen kenarlar her girdi turunda var, silinen kenarlar
#      hiçbirinde yok mu; komut satırından (ayrı süreç, duvar saati sınırıyla) aynı sonucu mu veriyor?
#   8) BetaTour çıkarımı ve ek kenarlar: modelin grafiği, modeli çalıştırmadan kurulan grafikle aynı mı; seçilen ek
#      kenarlar birleşimde yok, modelin grafiğinde var ve istenen sayıda mı?
def selfcheck(n=2000, trials=25, merge_trials=40):
    xy = T.to_int_coords(T.gen_instance(n, "clustered", np.random.default_rng(10093)))
    folder = WORK_ROOT / "selfcheck"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    problem, eax_problem = folder / "p.tsp", folder / "p_eax.tsp"
    T.write_tsp(str(problem), xy, name="selfcheck", ew="EUC_2D")
    B.write_eax_problem(str(eax_problem), xy)

    def lkh(binary, tag, extra=None, env=None, merge=(), max_trials=trials):
        par = folder / (tag + ".par")
        B.write_par(par, dict(PROBLEM_FILE=problem, RUNS=1, TRACE_LEVEL=2, MAX_TRIALS=max_trials, SEED=SEEDS[0],
                              **POP_PARAMS, **(extra or {}), OUTPUT_TOUR_FILE=folder / (tag + ".tour")),
                    merge_tour_files=merge)
        res = B.run_process([binary, par], folder, env=env, timeout=900)
        assert res["returncode"] == 0, res["stdout"][-500:]
        return res

    cand = folder / "cand.txt"
    plain = lkh(LKH, "plain")
    recorded = lkh(LKH_REC, "rec", extra={"CANDIDATE_FILE": cand}, env={"BETATOUR_TRIAL_DUMP": folder / "rec.bin"})
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
    deadline = dict(finished=bool(out["finished"]), stopped_by_deadline=bool(out["stopped_by_deadline"]),
                    valid=bool(B.is_permutation(tour, n) and int(T.tour_length(xy, tour)) == int(value)),
                    wall_s=float(out["wall_s"]))

    # LKH'nin kendi tur birleştirmesi: kayıtlı koşunun son farklı deneme turlarından en çok altısı birleştirilir.
    chosen, seen = [], set()
    for r in trial_rec[::-1]:
        key = B.edge_keys(B.tour_of(r), n).tobytes()
        if key not in seen:
            seen.add(key)
            chosen.append((int(r["cost"]), B.tour_of(r)))
        if len(chosen) == 6:
            break
    assert len(chosen) >= 2, "öz sınama: birleştirilecek iki farklı deneme turu yok"
    chosen.sort(key=lambda x: x[0])
    files = []
    for i, (cost, tour) in enumerate(chosen):
        files.append(folder / f"t{i}.tour")
        T.write_tour(str(files[-1]), tour, name=f"t{i}", length=cost)
    common = B.common_edge_keys([t for _, t in chosen], n)
    before = (cand.stat().st_mtime_ns, hashlib.sha256(cand.read_bytes()).hexdigest())
    extra = {"CANDIDATE_FILE": cand, "INITIAL_TOUR_FILE": files[0]}
    merge_plain = lkh(LKH, "merge_plain", extra=extra, merge=files, max_trials=merge_trials)
    merge_rec = lkh(LKH_REC, "merge_rec", extra=extra, merge=files, max_trials=merge_trials,
                    env={"BETATOUR_TRIAL_DUMP": folder / "merge.bin", "BETATOUR_BEST_ONLY": "1"})
    after = (cand.stat().st_mtime_ns, hashlib.sha256(cand.read_bytes()).hexdigest())
    merge_costs = B.lkh_trial_costs(merge_rec["stdout"])
    merged = B.read_records(folder / "merge.bin", n)
    final = T.read_tour(str(folder / "merge_rec.tour"))
    final_length = int(T.tour_length(xy, final))
    merge = dict(
        tours=int(len(chosen)), common_edges=int(len(common)), trials=int(merge_trials),
        same_search=bool(len(merge_costs) == merge_trials and merge_costs == B.lkh_trial_costs(merge_plain["stdout"])),
        best_only=bool(len(merged) >= 1 and set(merged["kind"].tolist()) == {1}),
        candidate_file_read=bool(before == after),
        valid=bool(B.is_permutation(final, n)), best_input=int(chosen[0][0]), final_length=final_length,
        not_longer=bool(final_length <= chosen[0][0]),
        keeps_common_edges=bool(np.isin(common, B.edge_keys(final, n), assume_unique=True).all()),
        last_record_is_lkh_tour=bool(len(merged) >= 1 and int(merged[-1]["cost"]) == final_length
                                     and np.array_equal(B.edge_keys(B.tour_of(merged[-1]), n), B.edge_keys(final, n))),
        wall_s=float(merge_rec["wall"]), pre_time_s=B.lkh_preprocessing_s(merge_rec["stdout"]))
    result = dict(
        nodes=n, trials=trials,
        same_search=bool(len(costs_plain) == trials and costs_plain == costs_rec),
        candidate_file_written=bool(cand.exists() and cand.stat().st_size > 0),
        trial_tours_recorded=int(len(trial_rec)), best_tours_recorded=int(len(best_rec)),
        recorded_tours_valid=bool(valid and list(trial_rec["trial"]) == list(range(1, trials + 1))),
        last_best_is_lkh_tour=bool(np.array_equal(B.edge_keys(own, n), B.edge_keys(B.tour_of(best_rec[-1]), n))
                                   and int(best_rec[-1]["cost"]) == min(c for _, c in costs_rec)),
        wall_plain_s=float(plain["wall"]), wall_recording_s=float(recorded["wall"]),
        best_input=int(min(costs)), eax=eax, deadline=deadline, merge=merge,
        union=union_check(folder, xy, [t for _, t in chosen]))
    shutil.rmtree(folder, ignore_errors=True)
    return result


def union_check(folder, xy, tours):
    n = len(xy)
    rng = np.random.default_rng(2727)
    small = []
    for _ in range(12):
        m = int(rng.integers(6, 10))
        sxy = rng.integers(0, 1000, size=(m, 2))
        stours = [rng.permutation(m) for _ in range(int(rng.integers(1, 4)))]
        skeys = U.union_keys(stours, m)
        stour, sinfo = U.solve_exact(sxy, skeys, min(stours, key=lambda t: U.tour_length(sxy, t)), 30.0)
        small.append(bool(sinfo["optimal"] and sinfo["length"] == U.brute_force(sxy, skeys)[0]))
    keys = U.union_keys(tours, n)
    lengths = [int(T.tour_length(xy, t)) for t in tours]
    start = tours[int(np.argmin(lengths))]
    red = U.reduce_graph(n, keys)
    consistent = True
    for t in tours:
        mask = np.isin(keys, U.tour_keys(t))
        consistent &= bool(np.all(mask[red["status"] == 1]) and not np.any(mask[red["status"] == 2]))
    tour, info = U.solve_exact(xy, keys, start, 120.0)
    src, dst = folder / "union_in.npz", folder / "union_out.npz"
    np.savez(src, xy=xy, keys=keys, start=start, deadline=np.float64(time.time() + 120.0), ew=np.array("EUC_2D"))
    res = B.run_process([sys.executable, UNION_PATH, "solve", src, dst], folder, timeout=300)
    with np.load(dst, allow_pickle=False) as z:
        cli_tour, cli_info = z["tour"].astype(np.int64), json.loads(str(z["info"]))
    succ = np.stack([T.tour_succ_pred(t)[0] for t in tours], axis=1).astype(np.int32)
    prediction = M.predict_tourfold(MODEL, xy, K_GRAPH, DEVICE, ens_succ=succ,
                                    ens_cost=np.array(lengths, dtype=np.float64))
    table = U.probability_table(prediction["prob"], prediction["NB"])
    k = int(round(EXTRA_FRACTION * n))
    top = U.extra_edges(table, keys, k, "probability")
    near = U.extra_edges(table, keys, k, "length", xy=xy)
    return dict(
        small_graphs_equal_enumeration=bool(all(small)), tours=len(tours), union_edges=int(len(keys)),
        fixed=info["fixed"], removed=info["removed"], core=info["core"], reduction_consistent=bool(consistent),
        optimal=bool(info["optimal"]), valid=bool(B.is_permutation(tour, n) and np.isin(U.tour_keys(tour), keys).all()),
        best_input=int(min(lengths)), length=int(info["length"]), not_longer=bool(info["length"] <= min(lengths)),
        length_recomputed=bool(int(T.tour_length(xy, tour)) == info["length"]), solve_s=info["solve_s"],
        cli_same=bool(res["returncode"] == 0 and cli_info["length"] == info["length"]
                      and int(T.tour_length(xy, cli_tour)) == cli_info["length"]),
        graph_same=bool(np.array_equal(model_graph(xy, succ), np.asarray(prediction["NB"], dtype=np.int64))),
        extra_ok=bool(len(top) == len(near) == k and not np.isin(top, keys).any() and not np.isin(near, keys).any()
                      and np.isin(top, table[0]).all() and np.isin(near, table[0]).all()),
        model_s=float(prediction["t_total"]), scip=SCIP_VERSION)


SELFCHECK = selfcheck(n=600, trials=12, merge_trials=20) if TEST_MODE else selfcheck()
SELFCHECK["session_id"] = SESSION_ID
write_json_atomic(OUTPUT_ROOT / "summary" / "selfcheck.json", SELFCHECK)
assert SELFCHECK["same_search"], "Kayıt eden LKH, değiştirilmemiş LKH ile aynı aramayı yapmadı."
assert SELFCHECK["candidate_file_written"], "Kayıt eden LKH aday dosyasını yazmadı."
assert SELFCHECK["recorded_tours_valid"] and SELFCHECK["last_best_is_lkh_tour"], "Kaydedilen turlar beklenen gibi değil."
for version, item in SELFCHECK["eax"].items():
    assert item["finished"] and item["valid"] and item["length_reported"] == item["length_recomputed"], \
        f"GA-EAX ({version}) geçerli ve uzunluğu doğru bir tur vermedi."
    assert item["length_reported"] <= SELFCHECK["best_input"], f"GA-EAX ({version}) en iyi girdi turundan uzun tur verdi."
    assert item["same_as_without_grid"], f"GA-EAX ({version}): ızgaralı komşu araması yamasız derlemeyle aynı sonucu vermedi."
assert SELFCHECK["deadline"]["finished"] and SELFCHECK["deadline"]["stopped_by_deadline"] and SELFCHECK["deadline"]["valid"], \
    "GA-EAX duvar saati sınırında en iyi turuyla durmadı."
MG = SELFCHECK["merge"]
assert MG["same_search"] and MG["best_only"], "LKH'nin tur birleştirmesi: kayıt eden kopya aramayı değiştirdi ya da deneme turlarını da yazdı."
assert MG["candidate_file_read"], "LKH'nin tur birleştirmesi aday dosyasını okumadı (dosya yeniden yazıldı)."
assert MG["valid"] and MG["not_longer"] and MG["last_record_is_lkh_tour"], "LKH'nin tur birleştirmesi beklenen turu vermedi."
assert MG["keeps_common_edges"], "LKH'nin tur birleştirmesi ortak kenarları korumadı."
X = SELFCHECK["union"]
assert X["small_graphs_equal_enumeration"], "Tam çözüm küçük graflarda sayımla aynı en kısa turu vermedi."
assert X["reduction_consistent"], "İndirgeme bir girdi turuyla çelişti (sabit kenar eksik ya da silinen kenar var)."
assert X["optimal"] and X["valid"] and X["not_longer"] and X["length_recomputed"], "Birleşim grafında tam çözüm beklenen turu vermedi."
assert X["cli_same"], "Komut satırından çalışan tam çözüm aynı sonucu vermedi."
assert X["graph_same"] and X["extra_ok"], "BetaTour grafiği ya da ek kenarlar beklenen gibi değil."
print("Öz sınama tamam. Kayıt eden LKH aynı aramayı yaptı (" + str(SELFCHECK["trials"]) + " deneme; süre "
      f"{SELFCHECK['wall_plain_s']:.1f} s / kayıtla {SELFCHECK['wall_recording_s']:.1f} s). GA-EAX: "
      + ", ".join(f"{v} {x['length_reported']} ({x['generations']} kuşak)" for v, x in SELFCHECK["eax"].items())
      + f"; en iyi girdi turu {SELFCHECK['best_input']}. LKH'nin tur birleştirmesi: {MG['tours']} tur, "
      f"{MG['common_edges']} ortak kenar korundu, {MG['best_input']} -> {MG['final_length']} ({MG['trials']} deneme). "
      f"Birleşimde tam çözüm: {X['tours']} tur, {X['union_edges']} kenar, çekirdek {X['core']} düğüm, "
      f"{X['best_input']} -> {X['length']} ({X['solve_s']:.1f} s, SCIP {X['scip']}).")
