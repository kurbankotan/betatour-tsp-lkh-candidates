POP_PARAMS = {"CANDIDATE_SET_TYPE": "POPMUSIC", "SUBGRADIENT": "NO", "MAX_CANDIDATES": K_LIST}
FRACTIONS = [0.25, 0.5, 0.75, 1.0]
SPLIT_TAG = f"{int(round(100 * SPLIT)):02d}"
# GA-EAX'in iki popülasyonu (run v26): "trials": her sürecin en iyi turu + son atılan deneme turları; "best6": yalnızca
# altı sürecin en iyi turları. "dual" kolu ikisini aynı anda iki çekirdekte koşturur ve kısa olan turu alır.
TRIALS_ARM, BEST6_ARM = "trials_" + SPLIT_TAG, "best6_" + SPLIT_TAG
DUAL_ARM, LKHMERGE_ARM = "dual_" + SPLIT_TAG, "lkhmerge_" + SPLIT_TAG
EAX_ARMS = [(TRIALS_ARM, "trials"), (BEST6_ARM, "best6")]
# Kademeli birleştirme (bu koşunun yeni kolu): aynı 102 tur. Aşama 1'de en iyi turlar dışındaki turlar GROUPS gruba
# bölünür ve GROUPS GA-EAX süreci aynı anda koşar; aşama 2'de grupların sonuç turları ve altı sürecin en iyi turları
# tek GA-EAX sürecinde birleştirilir.
STAGED_ARM = "staged_" + SPLIT_TAG
ARM_NAMES = ["lkh6", TRIALS_ARM, BEST6_ARM, DUAL_ARM, LKHMERGE_ARM, STAGED_ARM]
TOUR_ARMS = ["lkh6", TRIALS_ARM, BEST6_ARM, LKHMERGE_ARM, STAGED_ARM]


def write_json_atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, allow_nan=False))
    os.replace(tmp, path)


def write_npz_atomic(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def parallel(func, items):
    with ThreadPoolExecutor(max_workers=W) as pool:
        return list(pool.map(func, items))


def xy_hash(inst):
    return hashlib.sha256(np.ascontiguousarray(inst["xy"]).tobytes()).hexdigest()


def instance_files(inst):
    """Örneğin yerel diskteki klasörü, LKH'nin okuduğu TSPLIB dosyası ve GA-EAX'in okuduğu dosya."""
    folder = WORK_ROOT / "local" / inst["name"]
    folder.mkdir(parents=True, exist_ok=True)
    lkh_problem = inst["problem_file"] or str(folder / "p.tsp")
    if inst["problem_file"] is None and not Path(lkh_problem).exists():
        T.write_tsp(lkh_problem, inst["xy"], name=inst["name"], ew=inst["ew"])
    eax_problem = folder / "p_eax.tsp"
    if not eax_problem.exists():
        assert inst["ew"] == "EUC_2D"
        B.write_eax_problem(str(eax_problem), inst["xy"])
    return folder, lkh_problem, str(eax_problem)


def eax_version(n):
    return "Normal" if n <= EAX_NORMAL_MAX_N else "Large"


def tour_len(inst, tour):
    return int(T.tour_length(inst["xy"], tour, inst["ew"]))


def cut_times(budget):
    """LKH süreçlerinin durdurulduğu an: milisaniye (tam sayı) ve saniye."""
    cut_ms = int(math.floor(1000.0 * SPLIT * budget))
    return cut_ms, cut_ms / 1000.0


def checked_eax_tour(inst, out, best_file, label):
    """GA-EAX'in yazdığı tur: geçerli olmalı ve bildirdiği uzunluk kendi hesabımızla aynı olmalı."""
    n = len(inst["xy"])
    assert out["finished"], f"{label}: GA-EAX sonuç yazmadı (çıkış kodu {out['returncode']}): {out['tail']}"
    tour, value = B.read_eax_best(best_file, n)
    assert B.is_permutation(tour, n), f"{label}: GA-EAX turu geçersiz"
    assert tour_len(inst, tour) == value, f"{label}: GA-EAX'in bildirdiği uzunluk turun uzunluğuna eşit değil"
    return tour, value


def eax_record(out, value, version):
    return dict(version=version, generations=out["generations"], converged=bool(not out["stopped_by_deadline"]),
                stopped_by_deadline=bool(out["stopped_by_deadline"]), wall_s=float(out["wall_s"]),
                first_line_s=out["first_line_s"], last_line_s=out["last_line_s"], last_stage=out["last_stage"],
                curve=out["curve"], final_length=int(value), returncode=out["returncode"])


def make_recorded(inst):
    """Altı kayıt eden LKH süreci (POPMUSIC-5, kesintisiz; önceki koşulardaki pop_full kolunun ayarları). Her
    denemenin turu ve her yeni en iyi tur, duvar saati damgasıyla dosyaya yazılır; her süreç ayrıca aday listelerini
    LKH'nin kendi biçimiyle dosyaya yazar (LKH'nin kendi tur birleştirmesi bu dosyayı okur; arama değişmez, öz sınama
    denetler). Süreçler bütçeden STOP_PAD_S saniye sonra dışarıdan durdurulur; bütün zamanlar altı sürecin
    başlatıldığı andan itibaren duvar saatidir."""
    n, budget = len(inst["xy"]), inst["budget"]
    folder, lkh_problem, _ = instance_files(inst)
    rec_dir = folder / "rec"
    shutil.rmtree(rec_dir, ignore_errors=True)
    rec_dir.mkdir(parents=True)
    start = time.time()

    def one(seed):
        par = rec_dir / f"s{seed}.par"
        B.write_par(par, dict(PROBLEM_FILE=lkh_problem, RUNS=1, TRACE_LEVEL=1, TIME_LIMIT=round(1.25 * budget + 30.0, 1),
                              MAX_TRIALS=100000000, SEED=seed, **POP_PARAMS, CANDIDATE_FILE=rec_dir / f"cand_s{seed}.txt",
                              OUTPUT_TOUR_FILE=rec_dir / f"s{seed}.tour"))
        return B.run_process([LKH_REC, par], rec_dir, env={"BETATOUR_TRIAL_DUMP": rec_dir / f"s{seed}.bin"},
                             timeout=max(1.0, start + budget + STOP_PAD_S - time.time()))

    raw = parallel(one, SEEDS)
    wall = time.time() - start
    start_ms = int(round(start * 1000.0))
    budget_ms = int(math.floor(1000.0 * budget))
    runs, finals = [], []
    for seed, res in zip(SEEDS, raw):
        assert res["timed_out"] or res["returncode"] == 0, f"lkh6 / tohum {seed}: LKH çıkış kodu {res['returncode']}"
        assert (rec_dir / f"cand_s{seed}.txt").exists(), f"lkh6 / tohum {seed}: aday dosyası yazılmadı"
        records = B.read_records(rec_dir / f"s{seed}.bin", n)
        within = records[records["epoch_ms"] <= start_ms + budget_ms]
        best = within[within["kind"] == 1]
        assert len(best), f"lkh6 / tohum {seed}: bütçe içinde tur yok"
        tour = B.tour_of(best[-1])
        assert B.is_permutation(tour, n) and tour_len(inst, tour) == int(best[-1]["cost"]), \
            f"lkh6 / tohum {seed}: kaydedilen en iyi tur geçersiz"
        curve = B.improvements(records, start_ms)
        trial_end = (records["epoch_ms"][records["kind"] == 0] - start_ms) / 1000.0
        assert B.best_before(curve, budget_ms / 1000.0) == int(best[-1]["cost"])
        runs.append(dict(seed=int(seed), improvements=curve, best_at_budget=int(best[-1]["cost"]),
                         best_at_fraction=[B.best_before(curve, math.floor(1000.0 * f * budget) / 1000.0)
                                           for f in FRACTIONS],
                         trials_at_fraction=[int((trial_end <= math.floor(1000.0 * f * budget) / 1000.0).sum())
                                             for f in FRACTIONS],
                         pre_time_s=B.lkh_preprocessing_s(res["stdout"]), first_tour_s=curve[0][0],
                         last_record_s=float((int(records["epoch_ms"][-1]) - start_ms) / 1000.0),
                         stopped_from_outside=bool(res["timed_out"]), returncode=res["returncode"],
                         wall_s=float(res["wall"])))
        finals.append((int(best[-1]["cost"]), tour))
    at_fraction = []
    for j in range(len(FRACTIONS)):
        known = [r["best_at_fraction"][j] for r in runs if r["best_at_fraction"][j] is not None]
        at_fraction.append(min(known) if known else None)
    record = dict(runs=runs, best_at_budget=min(r["best_at_budget"] for r in runs),
                  mean_at_budget=float(np.mean([r["best_at_budget"] for r in runs])), best_at_fraction=at_fraction,
                  mean_trials_at_fraction=[float(np.mean([r["trials_at_fraction"][j] for r in runs]))
                                           for j in range(len(FRACTIONS))],
                  wall_s=float(wall), start_ms=start_ms)
    return record, {"lkh6": min(finals, key=lambda x: x[0])[1]}


def recorded_at_cut(inst, rec):
    """Kayıtlı koşunun dosyaları ve bütçenin SPLIT anındaki en iyi uzunluk (altı sürecin en iyisi)."""
    n, budget = len(inst["xy"]), inst["budget"]
    folder = instance_files(inst)[0]
    records = [B.read_records(folder / "rec" / f"s{seed}.bin", n) for seed in SEEDS]
    assert all(len(r) for r in records), "kayıt dosyaları eksik"
    cut_ms, cut_s = cut_times(budget)
    known = [B.best_before(run["improvements"], cut_s) for run in rec["runs"]]
    assert all(x is not None for x in known), \
        f"bütçenin %{100 * SPLIT:.0f} anında turu olmayan süreç var (ilk turlar: {[r['first_tour_s'] for r in rec['runs']]})"
    return records, cut_ms, cut_s, int(min(known))


def make_dual(inst, rec):
    """Birincil kol. Altı LKH süreci bütçenin SPLIT anında durur; kayıtlı turlarından iki popülasyon kurulur ve iki
    GA-EAX süreci aynı anda (iki çekirdekte) koşar. Her birine kalan süre, kendi popülasyonunu hazırlama süresi
    düşülerek verilir; sürecin kendi duvar saati sınırı vardır. Kolun sonucu iki turdan kısa olanıdır. Bir
    popülasyondaki turların hepsi aynıysa birleştirilecek bir şey yoktur: GA-EAX koşmaz, sonuç o turdur."""
    n, budget = len(inst["xy"]), inst["budget"]
    folder, _, eax_problem = instance_files(inst)
    records, cut_ms, cut_s, best_at_cut = recorded_at_cut(inst, rec)
    version = eax_version(n)
    jobs = []
    for name, kind in EAX_ARMS:
        begin = time.perf_counter()
        tours, costs, info = B.select_population(records, rec["start_ms"] + cut_ms, PER_PROCESS, n,
                                                 trial_tours=(kind == "trials"))
        pop_file = folder / (name + ".pop")
        B.write_population(pop_file, tours, costs)
        prep_s = time.perf_counter() - begin
        for tour, cost in zip(tours, costs):
            assert B.is_permutation(tour, n) and tour_len(inst, tour) == cost, f"{name}: kaydedilen tur geçersiz"
        assert min(costs) == best_at_cut, f"{name}: popülasyonun en iyi turu altı sürecin en iyi turu değil"
        jobs.append(dict(name=name, kind=kind, prep_s=float(prep_s), pop_file=pop_file, pop_size=len(tours),
                         best_tour=tours[int(np.argmin(costs))], per_process=info,
                         population=B.population_stats(tours, n)))

    def one(job):
        allowed = max(0.0, budget - cut_s - job["prep_s"])
        if job["population"]["distinct"] < 2:
            return allowed, None, None
        out, best_file = B.run_eax(EAX[version], eax_problem, str(folder / ("eax_" + job["name"])), "M",
                                   job["pop_size"], kids=EAX_KIDS, population_file=str(job["pop_file"]),
                                   deadline_epoch=time.time() + allowed)
        return allowed, out, best_file

    raw = parallel(one, jobs)
    arms, tours_out = {}, {}
    for job, (allowed, out, best_file) in zip(jobs, raw):
        if out is None:
            tour, at_budget, eax = job["best_tour"], best_at_cut, None
        else:
            tour, value = checked_eax_tour(inst, out, best_file, job["name"])
            at_deadline = B.best_before(out["curve"], allowed)
            at_budget = min(x for x in (best_at_cut, at_deadline) if x is not None)
            eax = eax_record(out, value, version)
        arms[job["name"]] = dict(kind=job["kind"], split=float(SPLIT), cut_s=float(cut_s), prep_s=job["prep_s"],
                                 eax_budget_s=float(allowed), pop_size=job["pop_size"], best_at_cut=best_at_cut,
                                 population=job["population"], per_process=job["per_process"], eax=eax,
                                 skipped=bool(out is None), at_budget=int(at_budget))
        tours_out[job["name"]] = tour
        Path(job["pop_file"]).unlink(missing_ok=True)
        shutil.rmtree(folder / ("eax_" + job["name"]), ignore_errors=True)
    shortest = min(arms[a]["at_budget"] for a, _ in EAX_ARMS)
    record = dict(arms=arms, at_budget=int(shortest), best_at_cut=best_at_cut,
                  source=[a for a, _ in EAX_ARMS if arms[a]["at_budget"] == shortest],
                  eax_processes=int(sum(not arms[a]["skipped"] for a, _ in EAX_ARMS)))
    return record, tours_out


def make_lkhmerge(inst, rec):
    """Kontrol: LKH-3'ün kendi tur birleştirmesi. Altı LKH süreci bütçenin SPLIT anında durur; altı en iyi turları
    MERGE_TOUR_FILE olarak altı yeni LKH sürecine verilir. LKH bu kipte altı turun ortak kenarlarını sabit tutar,
    turların kenarlarını aday listelerine ekler ve kalanını arar (IsCommonEdge.c, AddTourCandidates.c). Her süreç
    altı turun en iyisinden başlar (INITIAL_TOUR_FILE) ve kayıtlı koşudaki aynı sıradaki sürecin POPMUSIC-5 aday
    dosyasını okur (ön işleme yeniden yapılmaz). Süreçler kalan sürenin sonunda dışarıdan durdurulur; yalnızca süre
    içinde kaydedilen turlar sayılır. Altı tur aynıysa birleştirilecek bir şey yoktur: sonuç o turdur."""
    n, budget = len(inst["xy"]), inst["budget"]
    folder, lkh_problem, _ = instance_files(inst)
    rec_dir, mdir = folder / "rec", folder / "lkhmerge"
    records, cut_ms, cut_s, best_at_cut = recorded_at_cut(inst, rec)
    assert all((rec_dir / f"cand_s{seed}.txt").exists() for seed in SEEDS), "aday dosyaları eksik"
    shutil.rmtree(mdir, ignore_errors=True)
    mdir.mkdir(parents=True)
    begin = time.perf_counter()
    tours, costs, _ = B.select_population(records, rec["start_ms"] + cut_ms, 1, n, trial_tours=False)
    assert len(tours) == W and min(costs) == best_at_cut
    files = []
    for i, (tour, cost) in enumerate(zip(tours, costs)):
        assert B.is_permutation(tour, n) and tour_len(inst, tour) == cost, "lkhmerge: kaydedilen tur geçersiz"
        files.append(mdir / f"t{i}.tour")
        T.write_tour(str(files[-1]), tour, name=f"t{i}", length=cost)
    prep_s = time.perf_counter() - begin
    best_index = int(np.argmin(costs))
    common = B.common_edge_keys(tours, n)
    allowed = max(0.0, budget - cut_s - prep_s)
    record = dict(split=float(SPLIT), cut_s=float(cut_s), prep_s=float(prep_s), lkh_budget_s=float(allowed),
                  tours=len(tours), distinct=int(B.distinct_tours(tours, n)), common_edges=int(len(common)),
                  best_at_cut=best_at_cut)
    if record["distinct"] < 2:
        record.update(skipped=True, runs=[], processes_with_tour=0, at_deadline=None, at_budget=best_at_cut,
                      keeps_common_edges=None, wall_s=0.0)
        shutil.rmtree(mdir, ignore_errors=True)
        return record, {LKHMERGE_ARM: tours[best_index]}
    launch = time.time()
    launch_ms, allowed_ms = int(round(launch * 1000.0)), int(math.floor(1000.0 * allowed))

    def one(i):
        seed = MERGE_SEEDS[i]
        par = mdir / f"m{seed}.par"
        B.write_par(par, dict(PROBLEM_FILE=lkh_problem, RUNS=1, TRACE_LEVEL=2, TIME_LIMIT=round(1.25 * budget + 30.0, 1),
                              MAX_TRIALS=100000000, SEED=seed, **POP_PARAMS,
                              CANDIDATE_FILE=rec_dir / f"cand_s{SEEDS[i]}.txt", INITIAL_TOUR_FILE=files[best_index],
                              OUTPUT_TOUR_FILE=mdir / f"m{seed}.tour"), merge_tour_files=files)
        return B.run_process([LKH_REC, par], mdir,
                             env={"BETATOUR_TRIAL_DUMP": mdir / f"m{seed}.bin", "BETATOUR_BEST_ONLY": "1"},
                             timeout=max(1.0, launch + allowed + STOP_PAD_S - time.time()))

    raw = parallel(one, range(W))
    runs, best = [], None
    for i, res in enumerate(raw):
        seed = MERGE_SEEDS[i]
        found = B.read_records(mdir / f"m{seed}.bin", n)
        assert not len(found) or set(found["kind"].tolist()) == {1}, "lkhmerge: beklenmeyen kayıt türü"
        within = found[found["epoch_ms"] <= launch_ms + allowed_ms]
        curve = B.improvements(found, launch_ms)
        value, keeps = None, None
        if len(within):
            value, tour = int(within[-1]["cost"]), B.tour_of(within[-1])
            assert B.is_permutation(tour, n) and tour_len(inst, tour) == value, \
                f"lkhmerge / tohum {seed}: kaydedilen tur geçersiz"
            keeps = bool(np.isin(common, B.edge_keys(tour, n), assume_unique=True).all())
            if best is None or value < best[0]:
                best = (value, tour)
        failed = not (res["timed_out"] or res["returncode"] == 0) or value is None
        runs.append(dict(seed=int(seed), candidates_of_seed=int(SEEDS[i]), improvements=curve, best_in_time=value,
                         records=int(len(found)), trials=int(len(B.lkh_trial_costs(res["stdout"]))),
                         first_record_s=curve[0][0] if curve else None, keeps_common_edges=keeps,
                         stopped_from_outside=bool(res["timed_out"]), returncode=res["returncode"],
                         wall_s=float(res["wall"]), tail=res["stdout"][-300:] if failed else None))
    assert best is not None, "lkhmerge: hiçbir süreç süresi içinde tur kaydetmedi: " + str(runs[0]["tail"])
    at_budget = min(best_at_cut, best[0])
    record.update(skipped=False, runs=runs, processes_with_tour=int(sum(r["best_in_time"] is not None for r in runs)),
                  at_deadline=int(best[0]), at_budget=int(at_budget),
                  keeps_common_edges=bool(all(r["keeps_common_edges"] for r in runs if r["best_in_time"] is not None)),
                  wall_s=float(time.time() - launch))
    shutil.rmtree(mdir, ignore_errors=True)
    return record, {LKHMERGE_ARM: best[1] if best[0] <= best_at_cut else tours[best_index]}


def staged_groups(best_mask, groups):
    """Aşama 1'in grupları: en iyi turlar dışındaki turlar popülasyondaki sırayla gruplara dönüşümlü dağıtılır
    (popülasyon süreç süreç sıralı olduğundan her gruba her süreçten tur düşer)."""
    others = np.flatnonzero(~np.asarray(best_mask, dtype=bool))
    return [others[g::groups] for g in range(groups)]


def staged_merge(inst, workdir, eax_problem, tours, costs, best_mask, launch, allowed, version):
    """Kademeli birleştirme, [launch, launch + allowed] duvar saati aralığında. Aşama 1: GROUPS GA-EAX süreci aynı
    anda, her biri kendi grubunda, launch + STAGE1_SHARE x allowed anına kadar. Aşama 2: grupların sonuç turları ve en
    iyi turlar tek GA-EAX sürecinde, launch + allowed anına kadar. Bir sürecin sonucu, kendi sınırı içinde bulduğu en
    iyi tur olarak sayılır. Döndürür: kayıt, kaydedilecek tur (en kısa son tur) ve süre içindeki en iyi uzunluk."""
    n = len(inst["xy"])
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    groups = staged_groups(best_mask, GROUPS)
    stage1_end = launch + STAGE1_SHARE * allowed

    def one(g):
        idx = groups[g]
        sub, sc = [tours[i] for i in idx], [int(costs[i]) for i in idx]
        info = dict(size=int(len(idx)), distinct=int(B.distinct_tours(sub, n)), best_input=int(min(sc)))
        if info["distinct"] < 2:
            return info, None, None, sub[int(np.argmin(sc))]
        pop = workdir / f"g{g}.pop"
        B.write_population(pop, sub, sc)
        out, best_file = B.run_eax(EAX[version], eax_problem, str(workdir / f"g{g}"), "M", len(sub), kids=EAX_KIDS,
                                   population_file=str(pop), deadline_epoch=stage1_end)
        return info, out, best_file, None

    raw = parallel(one, range(GROUPS))
    stage1, candidates = [], []
    for g, (info, out, best_file, tour) in enumerate(raw):
        if out is None:
            info.update(eax=None, at_deadline=info["best_input"])
            candidates.append((info["best_input"], tour))
        else:
            tour, value = checked_eax_tour(inst, out, best_file, f"{STAGED_ARM} / grup {g}")
            info.update(eax=eax_record(out, value, version),
                        at_deadline=B.best_before(out["curve"], stage1_end - out["launch_epoch"]))
            candidates.append((int(value), tour))
        stage1.append(info)
    stage1_wall = time.time() - launch
    best_idx = np.flatnonzero(np.asarray(best_mask, dtype=bool))
    pool = candidates + [(int(costs[i]), tours[i]) for i in best_idx]
    stage2 = dict(pop_size=len(pool), distinct=int(B.distinct_tours([t for _, t in pool], n)), eax=None,
                  at_deadline=None, skipped=True, budget_s=float(max(0.0, launch + allowed - time.time())))
    if stage2["distinct"] >= 2 and stage2["budget_s"] >= 1.0:
        pop = workdir / "final.pop"
        B.write_population(pop, [t for _, t in pool], [c for c, _ in pool])
        out, best_file = B.run_eax(EAX[version], eax_problem, str(workdir / "final"), "M", len(pool), kids=EAX_KIDS,
                                   population_file=str(pop), deadline_epoch=launch + allowed)
        tour, value = checked_eax_tour(inst, out, best_file, f"{STAGED_ARM} / aşama 2")
        stage2.update(eax=eax_record(out, value, version), skipped=False,
                      at_deadline=B.best_before(out["curve"], launch + allowed - out["launch_epoch"]))
        pool.append((int(value), tour))
    best_at_cut = int(min(int(costs[i]) for i in best_idx))
    within = [x["at_deadline"] for x in stage1 if x["at_deadline"] is not None]
    if stage2["at_deadline"] is not None:
        within.append(int(stage2["at_deadline"]))
    at_budget = int(min([best_at_cut] + within))
    source = ("stage2" if stage2["at_deadline"] is not None and int(stage2["at_deadline"]) == at_budget
              else "stage1" if any(x["at_deadline"] == at_budget for x in stage1) else "cut")
    value, tour = min(pool, key=lambda x: x[0])
    assert value <= at_budget
    record = dict(stage1=stage1, stage1_wall_s=float(stage1_wall), stage2=stage2, at_budget=at_budget, source=source,
                  used_s=float(time.time() - launch))
    return record, tour, at_budget


def make_staged(inst, rec):
    """Kademeli birleştirme kolu. Altı LKH süreci bütçenin SPLIT anında durur; dual kolunun 102 turluk popülasyonu
    (süreç başına en iyi tur + son 16 deneme turu) kurulur ve staged_merge kalan süre içinde, kendi hazırlık süresi
    düşülerek, iki aşamada birleştirir. Aşama 1 GROUPS çekirdek, aşama 2 bir çekirdek kullanır."""
    n, budget = len(inst["xy"]), inst["budget"]
    folder, _, eax_problem = instance_files(inst)
    records, cut_ms, cut_s, best_at_cut = recorded_at_cut(inst, rec)
    version = eax_version(n)
    begin = time.perf_counter()
    tours, costs, info = B.select_population(records, rec["start_ms"] + cut_ms, PER_PROCESS, n, trial_tours=True)
    best_tours, _, _ = B.select_population(records, rec["start_ms"] + cut_ms, 1, n, trial_tours=False)
    best_keys = {B.edge_keys(t, n).tobytes() for t in best_tours}
    best_mask = np.array([B.edge_keys(t, n).tobytes() in best_keys for t in tours])
    prep_s = time.perf_counter() - begin
    for tour, cost in zip(tours, costs):
        assert B.is_permutation(tour, n) and tour_len(inst, tour) == cost, f"{STAGED_ARM}: kaydedilen tur geçersiz"
    assert min(costs) == best_at_cut and best_mask.any() and (~best_mask).sum() >= GROUPS
    allowed = max(0.0, budget - cut_s - prep_s)
    work = folder / "staged"
    shutil.rmtree(work, ignore_errors=True)
    staged, tour, at_budget = staged_merge(inst, work, eax_problem, tours, costs, best_mask, time.time(), allowed,
                                           version)
    record = dict(split=float(SPLIT), cut_s=float(cut_s), prep_s=float(prep_s), budget_s=float(allowed),
                  pop_size=len(tours), best_tours=int(best_mask.sum()), groups=GROUPS, stage1_share=float(STAGE1_SHARE),
                  per_process=info, population=B.population_stats(tours, n), best_at_cut=best_at_cut, **staged)
    shutil.rmtree(work, ignore_errors=True)
    return record, {STAGED_ARM: tour}


STEP_ORDERS = [("dual", "lkhmerge", "staged"), ("lkhmerge", "staged", "dual"), ("staged", "dual", "lkhmerge")]


def run_one_instance(inst, ordinal):
    name, n = inst["name"], len(inst["xy"])
    final_path = OUTPUT_ROOT / "instances" / (name + ".json")
    stage_dir = OUTPUT_ROOT / "stages" / name
    local = WORK_ROOT / "local" / name
    session_marker = stage_dir / "runtime_session_id.txt"
    if stage_dir.exists() and any(stage_dir.iterdir()):
        old_id = session_marker.read_text().strip() if session_marker.exists() else None
        if old_id != SESSION_ID:
            print("Yeni Colab makinesi: yarım kalan örnek baştan çalıştırılıyor:", name, flush=True)
            shutil.rmtree(stage_dir)
            shutil.rmtree(local, ignore_errors=True)
    stage_dir.mkdir(parents=True, exist_ok=True)
    session_marker.write_text(SESSION_ID)
    stage = {key: (stage_dir / f"{key}.json", stage_dir / f"tours_{key}.npz")
             for key in ("lkh6", "dual", "lkhmerge", "staged")}
    # Türev kolların sırası örnek numarasına göre döner (üç sıra); hepsi aynı kayıtlı koşudan çıkar.
    order = STEP_ORDERS[inst["index"] % len(STEP_ORDERS)]

    def save(key, record, tours):
        write_npz_atomic(stage[key][1], **{k: np.asarray(v).astype(np.int32) for k, v in tours.items()})
        write_json_atomic(stage[key][0], record)

    derived = {"dual": (make_dual, "iki GA-EAX süreci (" + ", ".join(a for a, _ in EAX_ARMS) + ")"),
               "lkhmerge": (make_lkhmerge, "LKH'nin kendi tur birleştirmesi"),
               "staged": (make_staged, f"kademeli birleştirme ({GROUPS} grup, sonra kazananlar)")}
    if not all(stage[key][0].exists() for key in stage):
        files = all((local / "rec" / f"s{seed}.bin").exists() and (local / "rec" / f"cand_s{seed}.txt").exists()
                    for seed in SEEDS)
        if not (stage["lkh6"][0].exists() and files):
            for key in derived:                # türev kolların hepsi aynı kayıtlı koşudan çıkmalıdır
                for path in stage[key]:
                    path.unlink(missing_ok=True)
            print(name, "/ altı kayıtlı LKH süreci", flush=True)
            save("lkh6", *make_recorded(inst))
        rec = json.loads(stage["lkh6"][0].read_text())
        for key in order:
            if not stage[key][0].exists():
                func, label = derived[key]
                print(name, "/", label, flush=True)
                save(key, *func(inst, rec))
        shutil.rmtree(local / "rec", ignore_errors=True)
    rec, dual, merge, staged = (json.loads(stage[key][0].read_text()) for key in ("lkh6", "dual", "lkhmerge", "staged"))
    results = {"lkh6": int(rec["best_at_budget"]), DUAL_ARM: int(dual["at_budget"]), LKHMERGE_ARM: int(merge["at_budget"]),
               STAGED_ARM: int(staged["at_budget"])}
    results.update({arm: int(dual["arms"][arm]["at_budget"]) for arm, _ in EAX_ARMS})
    assert set(results) == set(ARM_NAMES)
    assert results[DUAL_ARM] == min(results[arm] for arm, _ in EAX_ARMS)
    assert dual["best_at_cut"] == merge["best_at_cut"] == staged["best_at_cut"] >= rec["best_at_budget"]
    tours = {}
    for key in stage:
        with np.load(stage[key][1], allow_pickle=False) as z:
            tours.update({k: z[k] for k in z.files})
    assert set(tours) == set(TOUR_ARMS)
    final_lengths = {arm: tour_len(inst, tours[arm].astype(np.int64)) for arm in TOUR_ARMS}
    assert all(B.is_permutation(tours[arm].astype(np.int64), n) for arm in TOUR_ARMS), "kaydedilen tur geçersiz"
    assert all(final_lengths[arm] <= results[arm] for arm in TOUR_ARMS), "kaydedilen tur, kaydedilen sonuçtan uzun"
    write_npz_atomic(OUTPUT_ROOT / "tours" / (name + ".npz"), **tours)
    data = dict(schema=1, complete=True, name=name, group=inst["group"], family=inst["family"],
                index=int(inst["index"]), ordinal=int(ordinal), n=n, budget_s=float(inst["budget"]), ew=inst["ew"],
                xy_sha256=xy_hash(inst), seeds=SEEDS, merge_seeds=MERGE_SEEDS, fractions=FRACTIONS, split=float(SPLIT),
                order=list(order), hardware=HARDWARE, session_id=SESSION_ID, code_sha256=CODE_SHA256,
                results=results, final_tour_lengths=final_lengths, lkh6=rec, dual=dual, lkhmerge=merge, staged=staged,
                finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    write_json_atomic(final_path, data)
    shutil.rmtree(local, ignore_errors=True)
    print("TAMAM", name, "|", ", ".join(f"{arm}={results[arm]}" for arm in ARM_NAMES), flush=True)
    return data
