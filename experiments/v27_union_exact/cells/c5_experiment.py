POP_PARAMS = {"CANDIDATE_SET_TYPE": "POPMUSIC", "SUBGRADIENT": "NO", "MAX_CANDIDATES": K_LIST}
FRACTIONS = [0.25, 0.5, 0.75, 1.0]
SPLIT_TAG = f"{int(round(100 * SPLIT)):02d}"
# GA-EAX'in iki popülasyonu. "trials": her sürecin en iyi turu + son atılan deneme turları; "best6": yalnızca altı
# sürecin en iyi turları. Birincil kol (dual) ikisini aynı anda iki çekirdekte koşturur ve kısa olan turu alır.
TRIALS_ARM, BEST6_ARM = "trials_" + SPLIT_TAG, "best6_" + SPLIT_TAG
DUAL_ARM, LKHMERGE_ARM = "dual_" + SPLIT_TAG, "lkhmerge_" + SPLIT_TAG
EAX_ARMS = [(TRIALS_ARM, "trials"), (BEST6_ARM, "best6")]
# Birleşim grafında tam çözüm (SCIP): altı en iyi turun birleşimi; aynı birleşim + BetaTour'un en olası yeni kenarları;
# aynı birleşim + aynı sayıda en kısa yeni kenar (kontrol). Üçü de tek çekirdekte, aynı anda koşar.
EXACT_ARM, EXACT_BETA_ARM, EXACT_NEAR_ARM = "exact6_" + SPLIT_TAG, "exact6beta_" + SPLIT_TAG, "exact6near_" + SPLIT_TAG
EXACT_ARMS = [(EXACT_ARM, None), (EXACT_BETA_ARM, "probability"), (EXACT_NEAR_ARM, "length")]
ARM_NAMES = ["lkh6", TRIALS_ARM, BEST6_ARM, DUAL_ARM, LKHMERGE_ARM, EXACT_ARM, EXACT_BETA_ARM, EXACT_NEAR_ARM]
TOUR_ARMS = ["lkh6", TRIALS_ARM, BEST6_ARM, LKHMERGE_ARM, EXACT_ARM, EXACT_BETA_ARM, EXACT_NEAR_ARM]


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



def first_trial_tours(records, n):
    """Her sürecin ilk denemesinin turu (modelin okuduğu topluluk; run v4-v8 ve v21-v23'teki gibi M = 6)."""
    tours = []
    for r in records:
        first = r[(r["kind"] == 0) & (r["trial"] == 1)]
        assert len(first) == 1, "ilk deneme turu kayıtlı değil"
        tours.append(B.tour_of(first[0]))
    return tours


def model_graph(xy, succ):
    """BetaTour'un grafiği (16 en yakın komşu + modelin topluluk kenarları), model çalıştırılmadan; run v21'deki gibi."""
    xyf = np.asarray(xy, dtype=np.float64)
    span = float((xyf.max(0) - xyf.min(0)).max()) or 1.0
    xyn = (xyf - xyf.min(0)) / span
    nb, _ = M.knn_np(xyn, K_GRAPH)
    extra = int(MODEL.cfg.get("graph_extra", 0))
    if extra:
        nb, _, _ = M.extend_graph(xyn, nb, succ, extra)
    return np.asarray(nb, dtype=np.int64)


def make_exact(inst, rec):
    """Üç tam çözüm kolu. Altı LKH süreci bütçenin SPLIT anında durur; altı en iyi turun birleşim grafı (ve iki kolda
    ek kenarlar) SCIP ile tam çözülür (unionx_v27.solve_exact; başlangıç çözümü altı turun en iyisi). Her kol kendi
    sürecinde, tek çekirdekte, kendi duvar saati sınırıyla koşar; üçü aynı anda başlar. Bir kolun süresi bütçeden
    kesme anı ve kendi hazırlığı (birleşim; model kolunda BetaTour çıkarımı; kontrol kolunda modelin grafiği)
    düşülerek verilir. Ek kenar sayısı EXTRA_FRACTION x n; adaylar modelin grafiğindeki, birleşimde olmayan kenarlardır."""
    n, budget = len(inst["xy"]), inst["budget"]
    folder = instance_files(inst)[0]
    xdir = folder / "exact"
    shutil.rmtree(xdir, ignore_errors=True)
    xdir.mkdir(parents=True)
    records, cut_ms, cut_s, best_at_cut = recorded_at_cut(inst, rec)
    begin = time.perf_counter()
    tours, costs, _ = B.select_population(records, rec["start_ms"] + cut_ms, 1, n, trial_tours=False)
    assert len(tours) == W and min(costs) == best_at_cut
    for tour, cost in zip(tours, costs):
        assert B.is_permutation(tour, n) and tour_len(inst, tour) == cost, "exact: kaydedilen tur geçersiz"
    union = U.union_keys(tours, n)
    start = tours[int(np.argmin(costs))]
    common_s = time.perf_counter() - begin
    first = first_trial_tours(records, n)
    succ = np.stack([T.tour_succ_pred(t)[0] for t in first], axis=1).astype(np.int32)
    k_extra = int(round(EXTRA_FRACTION * n))
    jobs, graphs = [], {}
    for name, by in EXACT_ARMS:
        begin = time.perf_counter()
        model_info = None
        if by is None:
            keys, extra = union, np.zeros(0, dtype=np.int64)
        elif by == "probability":
            first_costs = np.array([T.tour_length(inst["xy"], t, inst["ew"]) for t in first], dtype=np.float64)
            prediction = M.predict_tourfold(MODEL, inst["xy"], K_GRAPH, DEVICE, ens_succ=succ, ens_cost=first_costs)
            graphs[by] = np.asarray(prediction["NB"], dtype=np.int64)
            table = U.probability_table(prediction["prob"], prediction["NB"])
            extra = U.extra_edges(table, union, k_extra, "probability")
            chosen_p = table[1][np.searchsorted(table[0], extra)]      # ek kenarlar modelin grafiğindedir
            keys = np.union1d(union, extra)
            model_info = dict(t_prep_s=float(prediction["t_prep"]), t_model_s=float(prediction["t_model"]),
                              graph_edges=int(len(table[0])), candidates=int((~np.isin(table[0], union)).sum()),
                              prob_min=float(chosen_p.min()) if len(chosen_p) else None,
                              prob_mean=float(chosen_p.mean()) if len(chosen_p) else None)
        else:
            nb = model_graph(inst["xy"], succ)
            graphs[by] = nb
            table = U.probability_table(np.zeros(nb.shape), nb)
            extra = U.extra_edges(table, union, k_extra, "length", xy=inst["xy"], ew=inst["ew"])
            keys = np.union1d(union, extra)
        in_file, out_file = xdir / f"{name}_in.npz", xdir / f"{name}_out.npz"
        np.savez(in_file, xy=inst["xy"], keys=keys, start=start, deadline=np.float64(0.0), ew=np.array(inst["ew"]))
        prep_s = common_s + (time.perf_counter() - begin)
        jobs.append(dict(name=name, by=by, keys=keys, extra=extra, prep_s=float(prep_s), model=model_info,
                         in_file=in_file, out_file=out_file))
    assert np.array_equal(graphs["probability"], graphs["length"]), "model grafiği ile kontrol grafiği farklı"
    launch = time.time()

    def one(job):
        allowed = max(0.0, budget - cut_s - job["prep_s"])
        deadline = launch + allowed
        with np.load(job["in_file"], allow_pickle=False) as z:
            data = {k: z[k] for k in z.files}
        data["deadline"] = np.float64(deadline)
        np.savez(job["in_file"], **data)
        res = B.run_process([sys.executable, UNION_PATH, "solve", job["in_file"], job["out_file"]], xdir,
                            timeout=allowed + EXACT_PAD_S)
        return allowed, deadline, res

    raw = parallel(one, jobs)
    arms, tours_out = {}, {}
    for job, (allowed, deadline, res) in zip(jobs, raw):
        assert res["returncode"] == 0 and Path(job["out_file"]).exists(), \
            f"{job['name']}: tam çözüm süreci başarısız (zaman aşımı: {res['timed_out']}): {res['stdout'][-1500:]}"
        with np.load(job["out_file"], allow_pickle=False) as z:
            tour, info = z["tour"].astype(np.int64), json.loads(str(z["info"]))
        keys = job["keys"]
        assert B.is_permutation(tour, n) and np.isin(U.tour_keys(tour), keys).all(), f"{job['name']}: tur geçersiz"
        length = tour_len(inst, tour)
        assert length == info["length"] and length <= best_at_cut, f"{job['name']}: uzunluk tutarsız"
        found_epoch = (info["solve_start_epoch"] + info["time_to_best_s"]
                       if info.get("solve_start_epoch") is not None and info.get("time_to_best_s") is not None else None)
        late = bool(found_epoch is not None and found_epoch > deadline + 1.0 and length < best_at_cut)
        tour_set = U.tour_keys(tour)
        arms[job["name"]] = dict(
            by=job["by"], split=float(SPLIT), cut_s=float(cut_s), prep_s=job["prep_s"], exact_budget_s=float(allowed),
            best_at_cut=best_at_cut, union_edges=int(len(union)), extra_edges=int(len(job["extra"])),
            extra_used=int(np.isin(job["extra"], tour_set).sum()), outside_union=int((~np.isin(tour_set, union)).sum()),
            model=job["model"], solver=info, finished_after_deadline_s=round(float(info["finished_epoch"] - deadline), 3),
            best_found_late=late, at_budget=int(best_at_cut if late else length))
        tours_out[job["name"]] = tour if not late else start
    record = dict(arms=arms, best_at_cut=best_at_cut, union_edges=int(len(union)), k_extra=k_extra,
                  pairwise=B.population_stats(tours, n), wall_s=float(time.time() - launch))
    tours_out["cut_tours"] = np.stack(tours).astype(np.int32)
    shutil.rmtree(xdir, ignore_errors=True)
    return record, tours_out


STEP_ORDERS = [("dual", "lkhmerge", "exact"), ("lkhmerge", "exact", "dual"), ("exact", "dual", "lkhmerge")]


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
             for key in ("lkh6", "dual", "lkhmerge", "exact")}
    # Türev kolların sırası örnek numarasına göre döner (üç sıra); hepsi aynı kayıtlı koşudan çıkar.
    order = STEP_ORDERS[inst["index"] % len(STEP_ORDERS)]

    def save(key, record, tours):
        write_npz_atomic(stage[key][1], **{k: np.asarray(v).astype(np.int32) for k, v in tours.items()})
        write_json_atomic(stage[key][0], record)

    derived = {"dual": (make_dual, "iki GA-EAX süreci (" + ", ".join(a for a, _ in EAX_ARMS) + ")"),
               "lkhmerge": (make_lkhmerge, "LKH'nin kendi tur birleştirmesi"),
               "exact": (make_exact, "birleşim grafında üç tam çözüm (" + ", ".join(a for a, _ in EXACT_ARMS) + ")")}
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
    rec, dual, merge, exact = (json.loads(stage[key][0].read_text()) for key in ("lkh6", "dual", "lkhmerge", "exact"))
    results = {"lkh6": int(rec["best_at_budget"]), DUAL_ARM: int(dual["at_budget"]), LKHMERGE_ARM: int(merge["at_budget"])}
    results.update({arm: int(dual["arms"][arm]["at_budget"]) for arm, _ in EAX_ARMS})
    results.update({arm: int(exact["arms"][arm]["at_budget"]) for arm, _ in EXACT_ARMS})
    assert set(results) == set(ARM_NAMES)
    assert results[DUAL_ARM] == min(results[arm] for arm, _ in EAX_ARMS)
    assert dual["best_at_cut"] == merge["best_at_cut"] == exact["best_at_cut"] >= rec["best_at_budget"]
    tours = {}
    for key in stage:
        with np.load(stage[key][1], allow_pickle=False) as z:
            tours.update({k: z[k] for k in z.files})
    assert set(tours) == set(TOUR_ARMS) | {"cut_tours"}
    final_lengths = {arm: tour_len(inst, tours[arm].astype(np.int64)) for arm in TOUR_ARMS}
    assert all(B.is_permutation(tours[arm].astype(np.int64), n) for arm in TOUR_ARMS), "kaydedilen tur geçersiz"
    assert all(final_lengths[arm] <= results[arm] for arm in TOUR_ARMS), "kaydedilen tur, kaydedilen sonuçtan uzun"
    union = U.union_keys(tours["cut_tours"].astype(np.int64), n)
    outside = {arm: int((~np.isin(U.tour_keys(tours[arm].astype(np.int64)), union)).sum()) for arm in TOUR_ARMS}
    write_npz_atomic(OUTPUT_ROOT / "tours" / (name + ".npz"), **tours)
    data = dict(schema=1, complete=True, name=name, group=inst["group"], family=inst["family"],
                index=int(inst["index"]), ordinal=int(ordinal), n=n, budget_s=float(inst["budget"]), ew=inst["ew"],
                xy_sha256=xy_hash(inst), seeds=SEEDS, merge_seeds=MERGE_SEEDS, fractions=FRACTIONS, split=float(SPLIT),
                order=list(order), hardware=HARDWARE, session_id=SESSION_ID, code_sha256=CODE_SHA256,
                union_sha256=UNION_SHA256, results=results, final_tour_lengths=final_lengths,
                edges_outside_union6=outside, lkh6=rec, dual=dual, lkhmerge=merge, exact=exact,
                finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    write_json_atomic(final_path, data)
    shutil.rmtree(local, ignore_errors=True)
    print("TAMAM", name, "|", ", ".join(f"{arm}={results[arm]}" for arm in ARM_NAMES), flush=True)
    return data
