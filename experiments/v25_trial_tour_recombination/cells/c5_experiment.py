POP_PARAMS = {"CANDIDATE_SET_TYPE": "POPMUSIC", "SUBGRADIENT": "NO", "MAX_CANDIDATES": K_LIST}
FRACTIONS = [0.25, 0.5, 0.75, 1.0]


def split_tag(f):
    return f"{int(round(100 * f)):02d}"


# Birleştirme kolları: (ad, popülasyon türü, bölme). "trials": her sürecin en iyi turu + son atılan deneme turları;
# "best6": yalnızca altı sürecin en iyi turları (atılan turların katkısını ayıran kontrol).
MERGE_ARMS = [(f"{kind}_{split_tag(f)}", kind, float(f)) for f in SPLITS for kind in ("trials", "best6")]
ARM_NAMES = ["lkh6"] + [name for name, _, _ in MERGE_ARMS] + ["cascade"]
PRIMARY_ARM = "trials_" + split_tag(PRIMARY_SPLIT)
CONTROL_ARM = "best6_" + split_tag(PRIMARY_SPLIT)


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
    denemenin turu ve her yeni en iyi tur, duvar saati damgasıyla dosyaya yazılır. Süreçler bütçeden STOP_PAD_S
    saniye sonra dışarıdan durdurulur; bütün zamanlar altı sürecin başlatıldığı andan itibaren duvar saatidir."""
    n, budget = len(inst["xy"]), inst["budget"]
    folder, lkh_problem, _ = instance_files(inst)
    rec_dir = folder / "rec"
    shutil.rmtree(rec_dir, ignore_errors=True)
    rec_dir.mkdir(parents=True)
    start = time.time()

    def one(seed):
        par = rec_dir / f"s{seed}.par"
        B.write_par(par, dict(PROBLEM_FILE=lkh_problem, RUNS=1, TRACE_LEVEL=1, TIME_LIMIT=round(1.25 * budget + 30.0, 1),
                              MAX_TRIALS=100000000, SEED=seed, **POP_PARAMS, OUTPUT_TOUR_FILE=rec_dir / f"s{seed}.tour"))
        return B.run_process([LKH_REC, par], rec_dir, env={"BETATOUR_TRIAL_DUMP": rec_dir / f"s{seed}.bin"},
                             timeout=max(1.0, start + budget + STOP_PAD_S - time.time()))

    raw = parallel(one, SEEDS)
    wall = time.time() - start
    start_ms = int(round(start * 1000.0))
    budget_ms = int(math.floor(1000.0 * budget))
    runs, finals = [], []
    for seed, res in zip(SEEDS, raw):
        assert res["timed_out"] or res["returncode"] == 0, f"lkh6 / tohum {seed}: LKH çıkış kodu {res['returncode']}"
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
    return record, min(finals, key=lambda x: x[0])[1]


def make_merges(inst, rec):
    """Kayıtlı koşunun turlarından popülasyonlar ve GA-EAX. Her bölme için LKH turları bütçenin o kesrindeki
    andan (duvar saati) alınır; GA-EAX'e kalan süre, popülasyonu hazırlama süresi düşülerek verilir. Altı GA-EAX
    süreci (üç bölme x iki popülasyon türü) paralel koşar; her birinin kendi duvar saati sınırı vardır."""
    n, budget = len(inst["xy"]), inst["budget"]
    folder, _, eax_problem = instance_files(inst)
    records = [B.read_records(folder / "rec" / f"s{seed}.bin", n) for seed in SEEDS]
    assert all(len(r) for r in records), "kayıt dosyaları eksik"
    version = eax_version(n)
    jobs = []
    for name, kind, f in MERGE_ARMS:
        cut_ms = int(math.floor(1000.0 * f * budget))
        cut_s = cut_ms / 1000.0
        begin = time.perf_counter()
        tours, costs, info = B.select_population(records, rec["start_ms"] + cut_ms, PER_PROCESS, n,
                                                 trial_tours=(kind == "trials"))
        pop_file = folder / (name + ".pop")
        B.write_population(pop_file, tours, costs)
        prep_s = time.perf_counter() - begin
        known = [B.best_before(run["improvements"], cut_s) for run in rec["runs"]]
        known = [x for x in known if x is not None]
        assert len(tours) >= 2 and known, f"{name}: bütçenin %{100 * f:.0f} anında birleştirilecek tur yok"
        for tour, cost in zip(tours, costs):
            assert B.is_permutation(tour, n) and tour_len(inst, tour) == cost, f"{name}: kaydedilen tur geçersiz"
        assert min(costs) == min(known), f"{name}: popülasyonun en iyi turu altı sürecin en iyi turu değil"
        jobs.append(dict(name=name, kind=kind, split=f, cut_s=float(cut_s), prep_s=float(prep_s), pop_file=pop_file,
                         pop_size=len(tours), best_at_cut=int(min(known)), per_process=info,
                         population=B.population_stats(tours, n)))

    def one(job):
        allowed = max(0.0, budget - job["cut_s"] - job["prep_s"])
        out, best_file = B.run_eax(EAX[version], eax_problem, str(folder / ("eax_" + job["name"])), "M",
                                   job["pop_size"], kids=EAX_KIDS, population_file=str(job["pop_file"]),
                                   deadline_epoch=time.time() + allowed)
        return allowed, out, best_file

    raw = parallel(one, jobs)
    merges, tours_out = {}, {}
    for job, (allowed, out, best_file) in zip(jobs, raw):
        tour, value = checked_eax_tour(inst, out, best_file, job["name"])
        at_deadline = B.best_before(out["curve"], allowed)
        at_budget = min(x for x in (job["best_at_cut"], at_deadline) if x is not None)
        merges[job["name"]] = dict(kind=job["kind"], split=job["split"], cut_s=job["cut_s"], prep_s=job["prep_s"],
                                   eax_budget_s=float(allowed), pop_size=job["pop_size"],
                                   best_at_cut=job["best_at_cut"], population=job["population"],
                                   per_process=job["per_process"], eax=eax_record(out, value, version),
                                   at_budget=int(at_budget))
        tours_out[job["name"]] = tour
        Path(job["pop_file"]).unlink(missing_ok=True)
    return merges, tours_out


def make_cascade(inst):
    """Bilinen melez: altı işçi paralel olarak LKH'nin tek denemelik turlarını üretir (her işçi ilk koşusunda
    POPMUSIC-5 aday dosyasını yazar, sonraki koşularında onu okur), sonra GA-EAX bu turlardan başlar ve bütçenin
    sonunda durur. Tur üretimi bütçenin CASCADE_GEN_FRACTION kesrinde ya da CASCADE_TOURS tura ulaşınca biter."""
    n, budget = len(inst["xy"]), inst["budget"]
    folder, lkh_problem, eax_problem = instance_files(inst)
    cdir = folder / "cascade"
    shutil.rmtree(cdir, ignore_errors=True)
    cdir.mkdir(parents=True)
    start = time.time()
    gen_limit = start + CASCADE_GEN_FRACTION * budget
    lock = threading.Lock()
    state = dict(started=0, failed=0)
    made = []

    def worker(w):
        cand = cdir / f"cand_{w}.txt"
        while True:
            with lock:
                if state["started"] >= CASCADE_TOURS or time.time() >= gen_limit:
                    return
                index = state["started"]
                state["started"] += 1
            par, tour_file = cdir / f"w{w}.par", cdir / f"w{w}.tour"
            tour_file.unlink(missing_ok=True)
            B.write_par(par, dict(PROBLEM_FILE=lkh_problem, RUNS=1, TRACE_LEVEL=1, MAX_TRIALS=1,
                                  SEED=CASCADE_SEED0 + index, **POP_PARAMS, CANDIDATE_FILE=cand,
                                  OUTPUT_TOUR_FILE=tour_file))
            res = B.run_process([LKH, par], cdir, timeout=max(1.0, start + budget - time.time()))
            if res["returncode"] != 0 or not tour_file.exists():
                with lock:
                    state["failed"] += 1
                    give_up = res["timed_out"] or state["failed"] > 3 * W
                if give_up:
                    return
                continue
            tour = T.read_tour(str(tour_file))
            with lock:
                made.append((index, tour, time.time() - start))

    parallel(worker, range(W))
    generation_s = time.time() - start
    made.sort(key=lambda x: x[0])
    tours = [tour for _, tour, _ in made]
    assert len(tours) >= 2, f"cascade: yalnızca {len(tours)} tur üretildi"
    begin = time.perf_counter()
    for tour in tours:
        assert B.is_permutation(tour, n), "cascade: LKH turu geçersiz"
    costs = [tour_len(inst, tour) for tour in tours]
    pop_file = cdir / "cascade.pop"
    B.write_population(pop_file, tours, costs)
    prep_s = time.perf_counter() - begin
    version = eax_version(n)
    eax_start_s = time.time() - start
    allowed = max(0.0, budget - eax_start_s)
    out, best_file = B.run_eax(EAX[version], eax_problem, str(cdir / "eax"), "C", len(tours), kids=EAX_KIDS,
                               population_file=str(pop_file), deadline_epoch=time.time() + allowed)
    tour, value = checked_eax_tour(inst, out, best_file, "cascade")
    at_deadline = B.best_before(out["curve"], allowed)
    at_budget = min(x for x in (min(costs), at_deadline) if x is not None)
    record = dict(tours=len(tours), failed_runs=int(state["failed"]), generation_s=float(generation_s),
                  prep_s=float(prep_s), eax_start_s=float(eax_start_s), eax_budget_s=float(allowed),
                  first_trial_best=int(min(costs)), first_trial_mean=float(np.mean(costs)),
                  tour_done_s=[round(float(t), 3) for _, _, t in made], population=B.population_stats(tours, n),
                  eax=eax_record(out, value, version), at_budget=int(at_budget),
                  wall_s=float(time.time() - start))
    shutil.rmtree(cdir, ignore_errors=True)
    return record, tour


def make_reference(batch):
    """Sıfırdan GA-EAX (yazarın varsayılan ayarları), örnek başına tek süreç; süre sınırı W x bütçe saniyedir
    (altı LKH sürecinin toplam süresi). Gruptaki örnekler paralel koşar (en çok W süreç, LKH kollarındaki yük)."""
    def one(inst):
        n, budget = len(inst["xy"]), inst["budget"]
        folder, _, eax_problem = instance_files(inst)
        version, limit = eax_version(n), W * budget
        out, best_file = B.run_eax(EAX[version], eax_problem, str(folder / "eax_scratch"), "R", EAX_SCRATCH_POP,
                                   kids=EAX_KIDS, deadline_epoch=time.time() + limit)
        tour, value = checked_eax_tour(inst, out, best_file, inst["name"] + " / referans")
        record = dict(schema=1, complete=True, name=inst["name"], group=inst["group"], n=n, budget_s=float(budget),
                      limit_s=float(limit), population=EAX_SCRATCH_POP, kids=EAX_KIDS, xy_sha256=xy_hash(inst),
                      eax=eax_record(out, value, version), at_limit=B.best_before(out["curve"], limit),
                      at_budget_wall=B.best_before(out["curve"], budget), parallel_runs=len(batch),
                      hardware=HARDWARE, session_id=SESSION_ID, code_sha256=CODE_SHA256,
                      finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        shutil.rmtree(folder / "eax_scratch", ignore_errors=True)
        return record, tour

    return parallel(one, batch)


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
    rec_path, merge_path, cascade_path = stage_dir / "lkh6.json", stage_dir / "merges.json", stage_dir / "cascade.json"
    # Kayıtlı LKH koşusu (ve birleştirmeleri) ile bilinen melezin sırası, aile içi sıra numarasına göre dönüşümlüdür.
    lkh_first = bool(inst["index"] % 2 == 0)

    def ensure_lkh_and_merges():
        if rec_path.exists() and merge_path.exists():
            return
        dumps = all((local / "rec" / f"s{seed}.bin").exists() for seed in SEEDS)
        if not (rec_path.exists() and dumps):
            print(name, "/ altı kayıtlı LKH süreci", flush=True)
            rec, tour = make_recorded(inst)
            write_npz_atomic(stage_dir / "tour_lkh6.npz", lkh6=tour.astype(np.int32))
            write_json_atomic(rec_path, rec)
        rec = json.loads(rec_path.read_text())
        print(name, "/ GA-EAX birleştirmeleri (" + ", ".join(a for a, _, _ in MERGE_ARMS) + ")", flush=True)
        merges, tours = make_merges(inst, rec)
        write_npz_atomic(stage_dir / "tours_merges.npz", **{k: v.astype(np.int32) for k, v in tours.items()})
        write_json_atomic(merge_path, merges)
        shutil.rmtree(local / "rec", ignore_errors=True)

    def ensure_cascade():
        if cascade_path.exists():
            return
        print(name, "/ cascade (bilinen melez)", flush=True)
        record, tour = make_cascade(inst)
        write_npz_atomic(stage_dir / "tour_cascade.npz", cascade=tour.astype(np.int32))
        write_json_atomic(cascade_path, record)

    for step in ([ensure_lkh_and_merges, ensure_cascade] if lkh_first else [ensure_cascade, ensure_lkh_and_merges]):
        step()
    rec, merges, cascade = (json.loads(p.read_text()) for p in (rec_path, merge_path, cascade_path))
    results = {"lkh6": int(rec["best_at_budget"]), "cascade": int(cascade["at_budget"])}
    results.update({arm: int(merges[arm]["at_budget"]) for arm, _, _ in MERGE_ARMS})
    assert set(results) == set(ARM_NAMES)
    assert all(merges[arm]["best_at_cut"] >= rec["best_at_budget"] for arm, _, _ in MERGE_ARMS)
    tours = {}
    for file in ("tour_lkh6.npz", "tours_merges.npz", "tour_cascade.npz"):
        with np.load(stage_dir / file, allow_pickle=False) as z:
            tours.update({k: z[k] for k in z.files})
    assert set(tours) == set(ARM_NAMES)
    final_lengths = {arm: tour_len(inst, tours[arm].astype(np.int64)) for arm in ARM_NAMES}
    assert all(final_lengths[arm] <= results[arm] for arm in ARM_NAMES), "kaydedilen tur, kaydedilen sonuçtan uzun"
    write_npz_atomic(OUTPUT_ROOT / "tours" / (name + ".npz"), **tours)
    data = dict(schema=1, complete=True, name=name, group=inst["group"], family=inst["family"],
                index=int(inst["index"]), ordinal=int(ordinal), n=n, budget_s=float(inst["budget"]), ew=inst["ew"],
                optimum=inst["optimum"], xy_sha256=xy_hash(inst), seeds=SEEDS, fractions=FRACTIONS,
                lkh_first=lkh_first, hardware=HARDWARE, session_id=SESSION_ID, code_sha256=CODE_SHA256,
                results=results, final_tour_lengths=final_lengths, lkh6=rec, merges=merges, cascade=cascade,
                finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    write_json_atomic(final_path, data)
    shutil.rmtree(local, ignore_errors=True)
    print("TAMAM", name, "|", ", ".join(f"{arm}={results[arm]}" for arm in ARM_NAMES), flush=True)
    return data
