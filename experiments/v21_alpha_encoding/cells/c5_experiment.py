POP_PARAMS = {"CANDIDATE_SET_TYPE": "POPMUSIC", "SUBGRADIENT": "NO", "MAX_CANDIDATES": K_LIST}
RESTART_ARMS = [arm for arm in ARM_NAMES if arm != "pop_full"]
FRACTIONS = [0.25, 0.5, 0.75, 1.0]
# Yeniden başlatılan her kolun aday dosyası: (aday dizisi, alfa dizisi, baba alanı yazılsın mı).
# Diziler candidates/<örnek>.npz içindedir; baba (Dad) alanı ilk aşamanın pop_dad dizisinden gelir.
ARM_FILES = {
    "pop_native": ("cand_pop", "alpha_pop_native", True),    # LKH'nin kendi dosyasının eşdeğeri
    "pop_rank": ("cand_pop", "alpha_pop_rank", False),       # run v5'teki restart_pop_raw dosyası
    "beta_tree": ("cand_beta", "alpha_beta_tree", True),     # BetaTour karma listesi, ağaç kodlaması
    "beta_rank": ("cand_beta", "alpha_beta_rank", False),    # run v5'teki betatour dosyası
    "freq_tree": ("cand_freq", "alpha_freq_tree", True),     # sayım karma listesi, ağaç kodlaması
}
assert set(ARM_FILES) == set(RESTART_ARMS)


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


def parallel_tasks(func, kwargs):
    with ThreadPoolExecutor(max_workers=W) as pool:
        futures = [pool.submit(func, **kw) for kw in kwargs]
        return [future.result() for future in futures]


def instance_common(inst):
    return dict(lkh_bin=LKH, workroot=str(WORK_ROOT / "lkh_tmp"), xy=inst["xy"],
                problem_file=inst["problem_file"], ew=inst["ew"])


def event_list(res, offset):
    return [[int(cost), float(t)] for cost, t in T.abs_events(res, offset=offset)]


def best_within(events, limit):
    costs = [int(cost) for cost, t in events if t <= limit + 1e-9]
    return min(costs) if costs else None


def run_record(label, seed, events, trial_end, budget, **extra):
    """Bir sürecin kaydı. Bütün zamanlar örneğin başlangıcından itibaren saniyedir (ortak saat)."""
    best = best_within(events, budget)
    assert best is not None, f"{label} / tohum {seed}: bütçe içinde tur yok"
    trial_end = [round(float(t), 3) for t in trial_end]
    return dict(seed=int(seed), events=events, best_at_budget=int(best),
                best_at_fraction=[best_within(events, f * budget) for f in FRACTIONS],
                trials_at_fraction=[int(sum(t <= f * budget + 1e-9 for t in trial_end)) for f in FRACTIONS],
                trial_end_s=trial_end, **extra)


def arm_record(runs, **extra):
    at_fraction = []
    for j in range(len(FRACTIONS)):
        known = [r["best_at_fraction"][j] for r in runs if r["best_at_fraction"][j] is not None]
        at_fraction.append(min(known) if known else None)
    return dict(runs=runs, best_at_budget=min(r["best_at_budget"] for r in runs),
                mean_at_budget=float(np.mean([r["best_at_budget"] for r in runs])),
                best_at_fraction=at_fraction,
                mean_trials_at_fraction=[float(np.mean([r["trials_at_fraction"][j] for r in runs]))
                                         for j in range(len(FRACTIONS))], **extra)


def exit_ok(res):
    return res["returncode"] == 0 or (res["returncode"] is None and res["timed_out"])


def make_full_arm(inst):
    """Kesintisiz POPMUSIC-5: altı süreç, bütçe sonunda dış saatle durdurulur (run v5'teki pop_full)."""
    budget = inst["budget"]
    common = instance_common(inst)
    start = time.time()
    raw = parallel_tasks(A.task_anytime, [dict(common, budget=budget, seed=seed, params=POP_PARAMS,
                                               timeout=budget + PAD_SECONDS) for seed in SEEDS])
    wall = time.time() - start
    runs = []
    for seed, res in zip(SEEDS, raw):
        assert exit_ok(res), f"pop_full / tohum {seed}: LKH çıkış kodu {res['returncode']}"
        assert res["pre_time"] is not None, f"pop_full / tohum {seed}: LKH ön işleme süresini yazmadı"
        offset = float(res["t_start"] - start)
        search_start = offset + float(res["pre_time"])
        runs.append(run_record("pop_full", seed, event_list(res, offset),
                               [search_start + t for t in res["trial_times"]], budget,
                               search_start_s=search_start, wall_s=float(res["wall"]),
                               returncode=res["returncode"], timed_out=bool(res["timed_out"])))
    return arm_record(runs, wall_s=float(wall))


def make_phase1(inst):
    """Ortak ilk aşama: her tohum için POPMUSIC-5 listeleri ve tek deneme. Süreç burada durur."""
    n = len(inst["xy"])
    common = instance_common(inst)
    start = time.time()
    raw = parallel_tasks(A.task_first_trial, [dict(common, seed=seed, k=K_LIST,
                                                    timeout=inst["budget"] + PAD_SECONDS) for seed in SEEDS])
    phase_wall = time.time() - start
    tours, cands, alphas, dads, costs, first_events = [], [], [], [], [], []
    for seed, res in zip(SEEDS, raw):
        assert T.is_valid_tour(res["tour"], n), f"ilk tur geçersiz: tohum {seed}"
        assert res["cand"].shape == (n, K_LIST) == res["alpha"].shape, f"aday boyutu yanlış: tohum {seed}"
        assert res["dad"].shape == (n,) and np.all(res["dad"] < n), f"baba alanı beklenmedik: tohum {seed}"
        events = event_list(res, res["t_start"] - start)
        assert len(events) == 1, f"ilk aşamada tek deneme bekleniyordu: tohum {seed}, {len(events)} kayıt"
        assert events[0][0] == int(round(res["cost"])), f"LKH'nin yazdığı maliyet turun uzunluğuna eşit değil: {seed}"
        assert events[0][1] <= inst["budget"], f"ilk deneme bütçeyi aştı: tohum {seed}"
        tours.append(res["tour"])
        cands.append(res["cand"])
        alphas.append(res["alpha"])
        dads.append(res["dad"])
        costs.append(res["cost"])
        first_events.append(events)
    arrays = dict(tours=np.stack(tours).astype(np.int32), pop_candidates=np.stack(cands).astype(np.int32),
                  pop_alpha=np.stack(alphas).astype(np.int64), pop_dad=np.stack(dads).astype(np.int32),
                  costs=np.asarray(costs, dtype=np.float64))
    meta = dict(wall_s=float(phase_wall), first_events=first_events, first_costs=[float(c) for c in costs],
                pre_time_s=[float(res["pre_time"] or 0.0) for res in raw],
                process_wall_s=[float(res["wall"]) for res in raw],
                nodes_with_father=[int((res["dad"] >= 0).sum()) for res in raw])
    return arrays, meta


def valid_candidates(cands, n):
    assert cands.shape == (W, n, K_LIST), cands.shape
    assert np.all((cands >= 0) & (cands < n)), "eksik veya geçersiz aday"
    assert not np.any(cands == np.arange(n)[None, :, None]), "düğüm kendi adayı olamaz"
    for i in range(K_LIST):
        for j in range(i + 1, K_LIST):
            assert not np.any(cands[:, :, i] == cands[:, :, j]), "tekrarlı aday"


def make_candidate_lists(inst, phase):
    """Aday dosyaları. Listeler run v5 ile aynıdır; yenilik dosyanın alfa ve baba (Dad) alanlarıdır.

    rank : alfa = 0, 100, 200, ..., baba alanı boş (run v1-v8'in yazdığı dosya; düğüm başına yalnızca ilk adayın
           alfası 0).
    tree : aynı tohumun kendi POPMUSIC listesinde alfası 0 olan kenarlar (LKH'nin en küçük 1-ağacı; düğüm başına
           yaklaşık iki aday) listede varsa alfa = 0, diğer girdiler liste sırasıyla 100, 200, ...; baba alanı
           aynı tohumun kendi dosyasından.
    """
    xy = inst["xy"]
    n = len(xy)
    tours, raw_pop, pop_alpha, costs = phase["tours"], phase["pop_candidates"], phase["pop_alpha"], phase["costs"]
    present = raw_pop >= 0
    assert raw_pop.shape == (W, n, K_LIST) == pop_alpha.shape
    assert np.all(present | (raw_pop == -1)) and np.all(raw_pop < n)
    assert not np.any(raw_pop == np.arange(n)[None, :, None])
    assert np.all(pop_alpha[present] >= 0) and np.all(np.diff(np.where(present, pop_alpha, 2**62), axis=2) >= 0)

    start = time.perf_counter()
    succ = np.stack([T.tour_succ_pred(tour)[0] for tour in tours], axis=1).astype(np.int32)
    successor_s = time.perf_counter() - start

    start = time.perf_counter()
    tree_keys = [A.tree_edge_keys(raw_pop[m], pop_alpha[m]) for m in range(W)]
    tree_keys_s = time.perf_counter() - start

    start = time.perf_counter()
    prediction = M.predict_tourfold(MODEL, xy, K_GRAPH, DEVICE, ens_succ=succ, ens_cost=costs)
    beta = np.stack([T.mix_candidates(prediction["rank"], raw_pop[m], k_first=K_LEARNED, k=K_LIST,
                                      fill=prediction["rank"][:, K_LEARNED:]) for m in range(W)]).astype(np.int32)
    beta_build_s = time.perf_counter() - start
    start = time.perf_counter()
    beta_tree = np.stack([A.tree_alpha(beta[m], tree_keys[m])[0] for m in range(W)])
    beta_encode_s = time.perf_counter() - start

    start = time.perf_counter()
    xyf = np.asarray(xy, dtype=np.float64)
    span = float((xyf.max(0) - xyf.min(0)).max()) or 1.0
    xyn = (xyf - xyf.min(0)) / span
    count_nb, count_d = M.knn_np(xyn, K_GRAPH)
    extra = int(MODEL.cfg.get("graph_extra", 0))
    if extra:
        count_nb, count_d, _ = M.extend_graph(xyn, count_nb, succ, extra)
    assert np.array_equal(count_nb, prediction["NB"]), "sayım ve model grafikleri farklı"
    freq_rank = T.frequency_candidates(count_nb, count_d, T.ensemble_frequency(count_nb, succ), k=K_GRAPH)
    freq = np.stack([T.mix_candidates(freq_rank, raw_pop[m], k_first=K_LEARNED, k=K_LIST,
                                      fill=freq_rank[:, K_LEARNED:]) for m in range(W)]).astype(np.int32)
    freq_build_s = time.perf_counter() - start
    start = time.perf_counter()
    freq_tree = np.stack([A.tree_alpha(freq[m], tree_keys[m])[0] for m in range(W)])
    freq_encode_s = time.perf_counter() - start

    for arr in (beta, freq):
        valid_candidates(arr, n)
    lists = dict(cand_pop=raw_pop.astype(np.int32), alpha_pop_native=pop_alpha.astype(np.int64),
                 alpha_pop_rank=np.stack([A.rank_alpha(raw_pop[m]) for m in range(W)]),
                 cand_beta=beta, alpha_beta_tree=beta_tree,
                 alpha_beta_rank=np.stack([A.rank_alpha(beta[m]) for m in range(W)]),
                 cand_freq=freq, alpha_freq_tree=freq_tree)

    # Tanılar (süreye dahil değildir). Ağaç kuralı LKH'nin kendi listesine uygulanınca LKH'nin alfa = 0 girdilerini
    # aynen vermelidir; uymayan girdi sayısı kaydedilir ve raporda gösterilir.
    rule_mismatch = [int((A.tree_alpha(raw_pop[m], tree_keys[m])[1] != (present[m] & (pop_alpha[m] == 0))).sum())
                     for m in range(W)]
    kept, zeros, tree_share = {}, {}, {}
    native_zero = float((present & (pop_alpha == 0)).sum())
    for arm, (cand_key, alpha_key, _) in ARM_FILES.items():
        cand, alpha = lists[cand_key], lists[alpha_key]
        kept[arm] = [A.kept_share(cand[m], alpha[m], tours[m]) for m in range(W)]
        zero = (cand >= 0) & (alpha == 0)
        zeros[arm] = float(zero.sum() / (W * n))
        tree_share[arm] = float(zero.sum() / native_zero)
    overhead = dict(pop_native=0.0, pop_rank=0.0,
                    beta_tree=successor_s + tree_keys_s + beta_build_s + beta_encode_s,
                    beta_rank=successor_s + beta_build_s,
                    freq_tree=successor_s + tree_keys_s + freq_build_s + freq_encode_s)
    meta = dict(overhead_s={arm: float(v) for arm, v in overhead.items()},
                successor_build_s=float(successor_s), tree_keys_s=float(tree_keys_s),
                beta_build_s=float(beta_build_s), beta_encode_s=float(beta_encode_s),
                freq_build_s=float(freq_build_s), freq_encode_s=float(freq_encode_s),
                model_predict_s=float(prediction["t_total"]),
                kept_share_by_seed=kept, kept_share={arm: float(np.mean(v)) for arm, v in kept.items()},
                zero_alpha_per_node=zeros, zero_alpha_relative_to_native=tree_share,
                tree_rule_mismatch=rule_mismatch,
                pop_candidates_per_node=float(present.sum() / (W * n)),
                beta_top3_in_pop5=float(np.mean([(beta[m, :, :K_LEARNED, None] == raw_pop[m, :, None, :]).any(-1).mean()
                                                 for m in range(W)])))
    return lists, meta


def make_resumed_arm(inst, phase, phase_meta, cand, alpha, dad, overhead_s, arm_name):
    """Ortak ilk aşamadan sonra yeniden başlatma: aday dosyası (cand, alpha, dad), sıfır ceza, ilk tur.

    Bütçeden ilk aşamanın duvar süresi ve listenin hazırlık süresi düşülür; kalan süre LKH'nin TIME_LIMIT değeridir."""
    budget = inst["budget"]
    prefix = float(phase_meta["wall_s"]) + float(overhead_s)
    remaining = budget - prefix
    firsts = phase_meta["first_events"]
    if remaining <= 0.25:
        runs = [run_record(arm_name, seed, firsts[m], [firsts[m][0][1]], budget, search_start_s=None, wall_s=0.0,
                           returncode=None, timed_out=False, cost_check=None) for m, seed in enumerate(SEEDS)]
        return arm_record(runs, continuation_wall_s=0.0, prelude_s=float(prefix),
                          continuation_budget_s=max(0.0, float(remaining)), skipped_due_budget=True)
    common = instance_common(inst)
    start = time.time()
    raw = parallel_tasks(A.task_guided, [dict(common, budget=remaining, seed=SEEDS[m], cand=cand[m], alpha=alpha[m],
                                              dad=None if dad is None else dad[m], init_tour=phase["tours"][m],
                                              timeout=remaining + PAD_SECONDS) for m in range(W)])
    wall = time.time() - start
    runs = []
    for m, (seed, res) in enumerate(zip(SEEDS, raw)):
        assert exit_ok(res), f"{arm_name} / tohum {seed}: LKH çıkış kodu {res['returncode']}"
        offset = prefix + float(res["t_start"] - start)
        resumed = event_list(res, offset)
        if res["cost_check"] is not None and res["returncode"] == 0:
            resumed.append([int(res["cost_check"]), float(offset + res["wall"])])
        search_start = offset + float(res["pre_time"] or 0.0)
        runs.append(run_record(arm_name, seed, sorted(firsts[m] + resumed, key=lambda e: e[1]),
                               [firsts[m][0][1]] + [search_start + t for t in res["trial_times"]], budget,
                               search_start_s=search_start, wall_s=float(res["wall"]), returncode=res["returncode"],
                               timed_out=bool(res["timed_out"]),
                               cost_check=None if res["cost_check"] is None else int(res["cost_check"])))
    return arm_record(runs, continuation_wall_s=float(wall), prelude_s=float(prefix),
                      continuation_budget_s=float(remaining), skipped_due_budget=False)


def load_phase(stage_dir):
    meta_path, array_path = stage_dir / "phase1.json", stage_dir / "phase1.npz"
    if meta_path.exists() and array_path.exists():
        with np.load(array_path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        return arrays, json.loads(meta_path.read_text())
    return None


def load_lists(stage_dir, name):
    meta_path, array_path = stage_dir / "candidates.json", OUTPUT_ROOT / "candidates" / (name + ".npz")
    if meta_path.exists() and array_path.exists():
        with np.load(array_path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        return arrays, json.loads(meta_path.read_text())
    return None


def xy_hash(inst):
    return hashlib.sha256(np.ascontiguousarray(inst["xy"]).tobytes()).hexdigest()


def run_one_instance(inst, ordinal):
    name = inst["name"]
    final_path = OUTPUT_ROOT / "instances" / (name + ".json")
    stage_dir = OUTPUT_ROOT / "stages" / name
    session_marker = stage_dir / "runtime_session_id.txt"
    if stage_dir.exists() and any(stage_dir.iterdir()):
        old_id = session_marker.read_text().strip() if session_marker.exists() else None
        if old_id != SESSION_ID:
            print("Yeni Colab makinesi: yarım kalan örnek baştan çalıştırılıyor:", name, flush=True)
            shutil.rmtree(stage_dir)
            (OUTPUT_ROOT / "candidates" / (name + ".npz")).unlink(missing_ok=True)
    stage_dir.mkdir(parents=True, exist_ok=True)
    session_marker.write_text(SESSION_ID)
    full_path = stage_dir / "pop_full.json"
    # Yeniden başlatılan kolların sırası örnek başına rastgeledir; kesintisiz kol, aile içi sıra numarası çiftse
    # önce, tekse sonra koşar (her ailede altı örnek önce, altı örnek sonra).
    control_order = [str(x) for x in np.random.default_rng(21000 + ordinal).permutation(RESTART_ARMS)]
    full_first = bool(inst["index"] % 2 == 0)

    def ensure_full():
        if not full_path.exists():
            print(name, "/ pop_full (kesintisiz)", flush=True)
            write_json_atomic(full_path, make_full_arm(inst))

    def ensure_controls():
        loaded = load_phase(stage_dir)
        if loaded is None:
            print(name, "/ ortak ilk denemeler", flush=True)
            phase, phase_meta = make_phase1(inst)
            write_npz_atomic(stage_dir / "phase1.npz", **phase)
            write_json_atomic(stage_dir / "phase1.json", phase_meta)
        else:
            phase, phase_meta = loaded
        loaded = load_lists(stage_dir, name)
        if loaded is None:
            print(name, "/ aday dosyaları", flush=True)
            lists, list_meta = make_candidate_lists(inst, phase)
            write_npz_atomic(OUTPUT_ROOT / "candidates" / (name + ".npz"), **lists)
            write_json_atomic(stage_dir / "candidates.json", list_meta)
        else:
            lists, list_meta = loaded
        for arm in control_order:
            path = stage_dir / (arm + ".json")
            if path.exists():
                continue
            print(name, "/", arm, flush=True)
            cand_key, alpha_key, with_dad = ARM_FILES[arm]
            write_json_atomic(path, make_resumed_arm(inst, phase, phase_meta, lists[cand_key], lists[alpha_key],
                                                     phase["pop_dad"] if with_dad else None,
                                                     list_meta["overhead_s"][arm], arm))
        return phase_meta, list_meta

    if full_first:
        ensure_full()
        phase_meta, list_meta = ensure_controls()
    else:
        phase_meta, list_meta = ensure_controls()
        ensure_full()
    arms = {arm: json.loads((stage_dir / (arm + ".json")).read_text()) for arm in ARM_NAMES}
    first_match = [int(arms["pop_full"]["runs"][m]["events"][0][0]) == int(phase_meta["first_events"][m][0][0])
                   for m in range(W)]
    data = dict(schema=1, complete=True, name=name, group=inst["group"], family=inst["family"],
                index=int(inst["index"]), ordinal=int(ordinal), n=len(inst["xy"]), budget_s=float(inst["budget"]),
                ew=inst["ew"], xy_sha256=xy_hash(inst), seeds=SEEDS, fractions=FRACTIONS,
                control_order=control_order, full_first=full_first, hardware=HARDWARE, session_id=SESSION_ID,
                code_sha256=CODE_SHA256, phase1=phase_meta, lists=list_meta,
                first_trial_matches_uninterrupted=first_match, arms=arms,
                finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    write_json_atomic(final_path, data)
    print("TAMAM", name, "|", ", ".join(f"{arm}={arms[arm]['best_at_budget']}" for arm in ARM_NAMES), flush=True)
    return data
