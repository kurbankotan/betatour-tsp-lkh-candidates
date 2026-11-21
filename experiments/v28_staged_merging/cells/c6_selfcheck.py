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
#   7) Kademeli birleştirme (yeni kol, deneydeki kodla): gruplar en iyi tur dışındaki her turu bir kez mi içeriyor;
#      iki aşama koşuyor mu; sonuç turu geçerli mi, en iyi girdi turundan uzun değil mi ve süre içindeki en iyi
#      uzunluktan kısa ya da ona eşit mi?
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
        staged=staged_check(folder, xy, eax_problem, records, int(records["epoch_ms"][-1]), trials))
    shutil.rmtree(folder, ignore_errors=True)
    return result


def staged_check(folder, xy, eax_problem, records, cut_ms, trials):
    """Kademeli birleştirme, deneydeki işlevlerle: kayıtlı koşunun en iyi turu ve deneme turları."""
    n = len(xy)
    tours, costs, _ = B.select_population([records], cut_ms, min(trials, 25), n)
    best, _, _ = B.select_population([records], cut_ms, 1, n, trial_tours=False)
    keys = {B.edge_keys(t, n).tobytes() for t in best}
    mask = np.array([B.edge_keys(t, n).tobytes() in keys for t in tours])
    covered = np.sort(np.concatenate(staged_groups(mask, GROUPS)))
    record, tour, at_budget = staged_merge(dict(xy=xy, ew="EUC_2D"), folder / "staged", str(eax_problem), tours, costs,
                                           mask, time.time(), 60.0, "Normal")
    length = int(T.tour_length(xy, tour))
    return dict(tours=len(tours), best_tours=int(mask.sum()), groups=GROUPS,
                groups_cover_others_once=bool(np.array_equal(covered, np.flatnonzero(~mask))),
                stage1_ran=int(sum(x["eax"] is not None for x in record["stage1"])),
                stage2_ran=bool(not record["stage2"]["skipped"]), valid=bool(B.is_permutation(tour, n)),
                best_input=int(min(costs)), length=length, not_longer=bool(length <= min(costs)),
                at_budget=int(at_budget), consistent=bool(length <= at_budget), used_s=float(record["used_s"]),
                source=record["source"])


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
S = SELFCHECK["staged"]
assert S["groups_cover_others_once"], "Kademeli birleştirme: gruplar en iyi tur dışındaki turları birer kez içermiyor."
assert S["stage1_ran"] >= 1 and S["stage2_ran"], "Kademeli birleştirme: aşamalar koşmadı."
assert S["valid"] and S["not_longer"] and S["consistent"], "Kademeli birleştirme beklenen turu vermedi."
print("Öz sınama tamam. Kayıt eden LKH aynı aramayı yaptı (" + str(SELFCHECK["trials"]) + " deneme; süre "
      f"{SELFCHECK['wall_plain_s']:.1f} s / kayıtla {SELFCHECK['wall_recording_s']:.1f} s). GA-EAX: "
      + ", ".join(f"{v} {x['length_reported']} ({x['generations']} kuşak)" for v, x in SELFCHECK["eax"].items())
      + f"; en iyi girdi turu {SELFCHECK['best_input']}. LKH'nin tur birleştirmesi: {MG['tours']} tur, "
      f"{MG['common_edges']} ortak kenar korundu, {MG['best_input']} -> {MG['final_length']} ({MG['trials']} deneme). "
      f"Kademeli birleştirme: {S['tours']} tur, {S['groups']} grup ({S['stage1_ran']} GA-EAX), "
      f"{S['best_input']} -> {S['length']} ({S['used_s']:.1f} s).")
