# Deneyin işlevleri: A) altı ilk deneme, B) BetaTour ve listeler, C) koşular. Her sonuç Drive'a hemen yazılır.
PRINT_LOCK = threading.Lock()
SESSION_ID = uuid.uuid4().hex[:12]
TMP = str(WORK_ROOT / "lkh_tmp")


def say(*parts):
    with PRINT_LOCK:
        print(f"[{time.strftime('%H:%M:%S')}]", *parts, flush=True)


def write_json_atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=float))
    os.replace(tmp, path)


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def save_npz_atomic(path, **arrays):
    path = Path(path)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def log_error(label, exc):
    write_json_atomic(OUTPUT_ROOT / "errors" / f"{label}_{int(time.time())}.json",
                      dict(label=label, error=repr(exc), traceback=traceback.format_exc(), session=SESSION_ID,
                           utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))


# ------------------------------------------------------------------------------------------ A: ilk denemeler
def first_done(inst):
    return (inst["dir"] / "first.npz").exists()


def stage_first(instances):
    """Her örnekte altı ilk deneme (POPMUSIC-5, tek deneme; pop kolunun aynı tohumlu ilk denemesi). Ana adımdaki
    koşularla aynı paralel yük altında ölçülür."""
    todo = [inst for inst in instances if not first_done(inst)]
    jobs = [(inst, j) for inst in todo for j in range(len(SEEDS))]
    results = {}
    with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
        futures = {pool.submit(V.first_trial, LKH, TMP, SEEDS[j], inst["xy"], inst["problem_file"], inst["ew"],
                               inst["optimum"], K_LIST): (inst["name"], j) for inst, j in jobs}
        for fut in as_completed(futures):
            name, j = futures[fut]
            results[(name, j)] = fut.result()
    for inst in todo:
        firsts = [results[(inst["name"], j)] for j in range(len(SEEDS))]
        for f in firsts:
            assert f["cost"] >= inst["optimum"], f"{inst['name']}: ilk deneme optimumdan kısa (optimum yanlış)"
        save_npz_atomic(inst["dir"] / "first.npz", tours=np.stack([f["tour"] for f in firsts]),
                        costs=np.array([f["cost"] for f in firsts]), cand=np.stack([f["cand"] for f in firsts]),
                        alpha=np.stack([f["alpha"] for f in firsts]), dad=np.stack([f["dad"] for f in firsts]),
                        cpu=np.array([f["time_total"] for f in firsts]), wall=np.array([f["wall"] for f in firsts]))
    return len(todo)


def load_firsts(inst):
    F = np.load(inst["dir"] / "first.npz")
    return [dict(tour=F["tours"][j], cost=float(F["costs"][j]), cand=F["cand"][j], alpha=F["alpha"][j],
                 dad=F["dad"][j], time_total=float(F["cpu"][j])) for j in range(len(SEEDS))]


# --------------------------------------------------------------------------------- B: BetaTour ve listeler
def stage_lists(inst):
    """BetaTour altı ilk deneme turunu okur; yeniden başlatılan kolların aday dosyaları kurulur ve her kolun duraklama
    süresi hesaplanır. Kapsama (pilotun optimal turu varsa): her listenin kaçırdığı optimal kenarlar."""
    path = inst["dir"] / "lists.npz"
    if path.exists():
        return "önbellekte"
    firsts = load_firsts(inst)
    succ = np.stack([T.tour_succ_pred(f["tour"])[0] for f in firsts], axis=1).astype(np.int32)
    costs = np.array([f["cost"] for f in firsts], dtype=np.float64)
    pred = M.predict_tourfold(MODEL, inst["xy"], K_GRAPH, DEVICE, ens_succ=succ, ens_cost=costs)
    lists, build, rank, count_rank = V.arm_lists(firsts, pred, k=K_LIST, k_union=K_UNION, k_learned=K_LEARNED)
    cpu = np.array([f["time_total"] for f in firsts])
    pause = V.pauses(cpu, pred["t_total"], build)
    arrays = {"rank": rank.astype(np.int32), "count_rank": count_rank.astype(np.int32),
              "NB": np.asarray(pred["NB"], dtype=np.int32), "prob": np.asarray(pred["prob"], dtype=np.float32)}
    for arm in V.RESTARTED:
        arrays[f"cand_{arm}"] = np.stack([c for c, _, _ in lists[arm]]).astype(np.int32)
        arrays[f"alpha_{arm}"] = np.stack([a for _, a, _ in lists[arm]]).astype(np.int64)
    arrays["dad"] = np.stack([f["dad"] for f in firsts]).astype(np.int32)
    meta = dict(pause=pause, build_s=build, model_s=float(pred["t_total"]), first_cpu=cpu.tolist(),
                first_costs=costs.tolist(), first_found=[bool(c == inst["optimum"]) for c in costs])
    if inst["opt_tour"] is not None:
        missed = {arm: [V.missed_optimal_edges(c, inst["opt_tour"]) for c, _, _ in lists[arm]] for arm in V.RESTARTED}
        missed["pop"] = list(missed["pop_r"])           # pop'un listeleri: aynı tohumun POPMUSIC-5 listeleri
        al = T.task_candidates(LKH, TMP, problem_file=inst["problem_file"], xy=inst["xy"], ew=inst["ew"],
                               ctype="ALPHA", subgradient=True, k=K_LIST)
        missed["alpha"] = [V.missed_optimal_edges(al["cand"], inst["opt_tour"])] * len(SEEDS)
        meta["missed_optimal_edges"] = missed
    save_npz_atomic(path, **arrays)
    write_json_atomic(inst["dir"] / "lists.json", meta)
    return f"BetaTour {pred['t_total']:.2f} s, duraklama {pause['beta']:.1f} s"


LIST_CACHE, LIST_LOCK = {}, threading.Lock()


def lists_of(inst):
    with LIST_LOCK:
        if inst["name"] not in LIST_CACHE:
            Z = np.load(inst["dir"] / "lists.npz")
            LIST_CACHE[inst["name"]] = ({key: Z[key] for key in Z.files}, read_json(inst["dir"] / "lists.json"),
                                        np.load(inst["dir"] / "first.npz")["tours"])
        return LIST_CACHE[inst["name"]]


# ------------------------------------------------------------------------------------------------ C: koşular
def run_path(inst, arm, j):
    return OUTPUT_ROOT / "runs" / inst["name"] / f"{arm}_{SEEDS[j]}.json"


def run_job(inst, arm, j):
    """Bir koşu: optimumda ya da üst sınırda durur. Kayıt hemen Drive'a yazılır."""
    path = run_path(inst, arm, j)
    if path.exists():
        return read_json(path)
    cap = inst["cap"]
    rec = dict(instance=inst["name"], group=inst["group"], n=inst["n"], arm=arm, seed=SEEDS[j], cap=cap,
               optimum=inst["optimum"], session=SESSION_ID, cpu_model=HARDWARE["cpu_model"])
    if arm in V.UNINTERRUPTED:
        res = V.lkh_run(LKH, TMP, inst["problem_file"], inst["xy"], inst["ew"], SEEDS[j], inst["optimum"], cap,
                        params=dict(V.UNINTERRUPTED[arm], MAX_CANDIDATES=K_LIST))
        rec.update(res, pause=0.0, first_found=False)
    else:
        arrays, meta, tours = lists_of(inst)
        rec.update(pause=float(meta["pause"][arm]), first_cpu=float(meta["first_cpu"][j]),
                   first_found=bool(meta["first_found"][j]))
        remaining = cap - rec["pause"]
        if rec["first_found"]:
            rec.update(found=True, time_total=None, note="first trial reached the optimum")
        elif remaining <= 0.05:
            rec.update(found=False, time_total=None, note="pause longer than the cap")
        else:
            files = (arrays[f"cand_{arm}"][j], arrays[f"alpha_{arm}"][j], arrays["dad"][j])
            res = V.lkh_run(LKH, TMP, inst["problem_file"], inst["xy"], inst["ew"], SEEDS[j], inst["optimum"],
                            remaining, files=files, init_tour=tours[j])
            rec.update(res)
    if rec.get("shorter") or rec.get("new_optimum"):
        raise RuntimeError(f"{inst['name']} {arm} {SEEDS[j]}: optimumdan kısa tur bulundu; optimum değeri yanlış")
    if not rec["first_found"] and rec.get("time_total") is None and rec.get("note") is None:
        rec.update(found=False, note=f"LKH did not report Time.total (rc={rec.get('returncode')}, "
                                     f"timed out={rec.get('timed_out')})")
        log_error(f"{inst['name']}_{arm}_{SEEDS[j]}", RuntimeError(rec["note"] + " " + rec.get("stdout_tail", "")[-400:]))
    tto = V.time_to_optimum(rec, cap)
    rec["tto"] = tto if math.isfinite(tto) else None
    rec["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json_atomic(path, rec)
    return rec


def all_jobs():
    jobs = [(inst["name"], arm, j) for inst in INSTANCES for arm in ARMS for j in range(len(SEEDS))]
    order = np.random.default_rng(JOB_ORDER_SEED).permutation(len(jobs))
    return [jobs[i] for i in order]
