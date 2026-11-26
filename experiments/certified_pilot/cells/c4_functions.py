# İşlevler: örnekler, LKH turları ve listeleri, BetaTour, Concorde (kök LP ve kanıtlı optimum), örnek başına analiz.
# Her adım sonucunu Drive'a yazar; oturum düşerse defter yeniden çalıştırıldığında biten adımlar atlanır.
PRINT_LOCK = threading.Lock()
T0 = time.time()


def say(*parts):
    with PRINT_LOCK:
        print(f"[{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:5.1f} dk]", *parts, flush=True)


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=float))
    os.replace(tmp, path)


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def save_npz(path, **arrays):
    path = Path(path)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def make_instances():
    """Yeni sentetik örnekler (çalışmanın üreticisi) ve optimumu bilinen TSPLIB örnekleri."""
    out = []
    for group, n, count, family, seed in SYNTHETIC:
        rng = np.random.default_rng(seed)
        for i in range(count):
            out.append(dict(group=group, family=family, name=f"cp1_{group}_{i:02d}", n=n,
                            xy=T.to_int_coords(T.gen_instance(n, family, rng)), ew="EUC_2D", known_opt=None,
                            source=None))
    if TEST_MODE:                       # TSPLIB biçimli küçük dosya, ondalıklı koordinatlarla; ağ gerekmez
        folder = WORK_ROOT / "tsplib_test"
        folder.mkdir(exist_ok=True)
        paths = {}
        for name in TSPLIB_NAMES:
            count = int(name[len("test_euc"):])
            xy = np.random.default_rng(73 + count).random((count, 2)) * 5000.0
            paths[name] = str(folder / (name + ".tsp"))
            with open(paths[name], "w") as f:
                f.write(f"NAME : {name}\nTYPE : TSP\nDIMENSION : {count}\nEDGE_WEIGHT_TYPE : EUC_2D\n"
                        "NODE_COORD_SECTION\n")
                f.writelines(f"{i} {x:.3f} {y:.3f}\n" for i, (x, y) in enumerate(xy, 1))
                f.write("EOF\n")
    else:
        paths = T.download_tsplib(str(WORK_ROOT / "tsplib"), names=TSPLIB_NAMES)
        absent = sorted(set(TSPLIB_NAMES) - set(paths))
        if absent:
            print("UYARI: TSPLIB dosyaları indirilemedi, bu örnekler atlanıyor:", absent)
    for name in TSPLIB_NAMES:
        if name not in paths:
            continue
        data = T.read_tsplib(paths[name])
        assert data["edge_weight_type"] == "EUC_2D" and len(data["xy"]) == data["dimension"], name
        out.append(dict(group="tsplib", family="tsplib", name=name, n=int(data["dimension"]), xy=data["xy"],
                        ew="EUC_2D", known_opt=T.TSPLIB_OPT.get(name), source=paths[name]))
    problems = WORK_ROOT / "problems"
    problems.mkdir(exist_ok=True)
    for inst in out:
        inst["dir"] = OUTPUT_ROOT / "instances" / inst["name"]
        inst["dir"].mkdir(parents=True, exist_ok=True)
        pf = problems / f"{inst['name']}.tsp"
        if inst["source"] is None:
            T.write_tsp(pf, inst["xy"], name=inst["name"], ew="EUC_2D")
        else:
            shutil.copy2(inst["source"], pf)
        inst["problem_file"] = str(pf)
        digest = hashlib.sha256(pf.read_bytes()).hexdigest()
        meta = inst["dir"] / "instance.json"
        if meta.exists():
            assert read_json(meta)["problem_sha256"] == digest, f"{inst['name']}: örnek dosyası değişti"
        else:
            write_json(meta, dict(name=inst["name"], group=inst["group"], family=inst["family"], n=inst["n"],
                                  known_opt=inst["known_opt"], problem_sha256=digest))
    assert len({x["name"] for x in out}) == len(out)
    return out


# ------------------------------------------------------------------------------------- A: LKH turları ve listeleri
def stage_lkh(inst):
    """Altı ilk deneme turu ve POPMUSIC-5 listeleri (BetaTour'un okuduğu topluluk), LKH'nin alfa-5 listesi ve düğüm
    cezaları (alt gradyan yükselişi; C1 sertifikası bunları kullanır) ve bir üst sınır turu."""
    path = inst["dir"] / "lkh.npz"
    if path.exists():
        return "önbellekte"
    n, xy, pf, ew = inst["n"], inst["xy"], inst["problem_file"], inst["ew"]
    work = WORK_ROOT / "lkh_runs" / inst["name"]
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    firsts = [T.task_first_trial(LKH, str(work), seed, xy, problem_file=pf, ew=ew, ctype="POPMUSIC", k=K_LIST,
                                 timeout=900) for seed in FIRST_TRIAL_SEEDS]
    t1 = time.time()
    al = T.lkh_solve(LKH, problem_file=pf, ew=ew, workroot=str(work), want_cands=True, want_pi=True, timeout=1800,
                     params={"CANDIDATE_SET_TYPE": "ALPHA", "SUBGRADIENT": "YES", "MAX_CANDIDATES": K_LIST,
                             "MAX_TRIALS": 1, "SEED": 1})
    assert "cand" in al and "pi" in al, "LKH aday ya da ceza dosyası yazmadı: " + al["stdout"][-500:]
    t2 = time.time()
    ub = T.lkh_solve(LKH, problem_file=pf, ew=ew, workroot=str(work), timeout=1800,
                     params={"CANDIDATE_SET_TYPE": "POPMUSIC", "SUBGRADIENT": "NO", "MAX_CANDIDATES": K_LIST,
                             "MAX_TRIALS": n, "SEED": 1, "TIME_LIMIT": max(5.0, UB_TIME_PER_NODE * n)})
    assert T.is_valid_tour(ub["tour"], n), "üst sınır koşusu geçerli tur vermedi"
    tours = [(T.tour_length(xy, ub["tour"], ew), np.asarray(ub["tour"]))] + [(f["cost"], f["tour"]) for f in firsts]
    ub_cost, ub_tour = min(tours, key=lambda x: x[0])
    t3 = time.time()
    save_npz(path, first_tours=np.stack([f["tour"] for f in firsts]).astype(np.int32),
             first_costs=np.array([f["cost"] for f in firsts]), pop_cand=np.stack([f["cand"] for f in firsts]),
             alpha_cand=T.pad_cols(al["cand"]["cand"], K_LIST), alpha_vals=T.pad_cols(al["cand"]["alpha"], K_LIST),
             pi=np.asarray(al["pi"], dtype=np.int64),
             lkh_lb=np.float64(al["lower_bound"] if al["lower_bound"] is not None else np.nan),
             ub_tour=np.asarray(ub_tour, dtype=np.int32), ub_cost=np.float64(ub_cost),
             times=np.array([t1 - t0, t2 - t1, t3 - t2]))
    shutil.rmtree(work, ignore_errors=True)
    return f"LKH {t3 - t0:.0f} s"


# ------------------------------------------------------------------------------------------------- B: BetaTour
def stage_model(inst):
    path = inst["dir"] / "model.npz"
    if path.exists():
        return "önbellekte"
    L = np.load(inst["dir"] / "lkh.npz")
    succ = np.stack([T.tour_succ_pred(t)[0] for t in L["first_tours"]], axis=1).astype(np.int32)
    pred = M.predict_tourfold(MODEL, inst["xy"], K_GRAPH, DEVICE, ens_succ=succ,
                              ens_cost=L["first_costs"].astype(np.float64))
    save_npz(path, NB=np.asarray(pred["NB"], dtype=np.int32), prob=np.asarray(pred["prob"], dtype=np.float32),
             rank=np.asarray(pred["rank"], dtype=np.int32), t_prep=np.float64(pred["t_prep"]),
             t_model=np.float64(pred["t_model"]))
    return f"BetaTour {pred['t_total']:.1f} s"


# ------------------------------------------------------------------------------------------------- C: Concorde
def full_edge_file(inst):
    path = WORK_ROOT / "full_edges" / f"{inst['name']}.edg"
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        C.write_full_edge_file(path, C.distance_matrix(inst["xy"], inst["ew"]))
    return path


def concorde_root(inst, z, full_e, tag):
    """Kök LP (dallanmadan; -B), z > 0 ise düğüm başına en küçük z indirgenmiş maliyet dökümüyle (-z z)."""
    args = ["-B", "-x", "-s", CONCORDE_SEED]
    if z:
        args += ["-z", int(z)]
    if full_e:
        args += ["-E", str(full_edge_file(inst))]
    wd = WORK_ROOT / "concorde_runs" / inst["name"] / tag
    res = C.run_concorde(CONCORDE, inst["problem_file"], wd, args, ROOT_TIMEOUT_S)
    shutil.copy2(wd / "stdout.txt", OUTPUT_ROOT / "logs" / f"{inst['name']}_{tag}.txt")
    res["lb"] = C.root_lower_bound(res)
    dump = None
    rcn = C.find_rcn(wd) if z else None
    if rcn is not None and res["lb"] is not None:
        dump = C.read_rcn(rcn, inst["n"])
    shutil.rmtree(wd, ignore_errors=True)
    return res, dump


def job_root(inst):
    path = inst["dir"] / "concorde_root.json"
    if path.exists():
        return "önbellekte"
    n = inst["n"]
    ub_lkh = float(np.load(inst["dir"] / "lkh.npz")["ub_cost"])
    z = 0 if C2_MODE is None else (min(K_RC, n - 1) if C2_MODE["partial"] else n - 1)
    res, dump = concorde_root(inst, z, bool(C2_MODE and C2_MODE["full_e"]), "root")
    rec = {k: v for k, v in res.items() if k != "workdir"}
    rec.update(ub_lkh=ub_lkh, z=z, dump=None)
    if dump is not None:
        lb = res["lb"]
        complete = bool(len(dump["keys"]) == n * (n - 1) // 2)
        thr = C.node_thresholds(dump["keys"], dump["rc"], n, z) if not complete else np.full(n, np.inf)
        cap = (ub_lkh - lb) + max(1.0, 1e-6 * abs(lb)) + max(RC_TOL, dump["quantum"])
        keep = dump["rc"] <= cap
        save_npz(inst["dir"] / "rc_keep.npz", keys=dump["keys"][keep], rc=dump["rc"][keep], thr=thr)
        rec["dump"] = dict(lines=dump["lines"], edges=int(len(dump["keys"])), complete=complete,
                           min_count=int(dump["counts"].min()), dup_max_diff=dump["dup_max_diff"],
                           sum_negative=float(np.clip(-dump["rc"], 0, None).sum()),
                           negative=int((dump["rc"] < -RC_TOL).sum()), min_rc=float(dump["rc"].min()), cap=cap,
                           stored=int(keep.sum()), quantum=dump["quantum"])
    write_json(path, rec)
    return f"kök LP {res['wall']:.0f} s" + (" (süre aşıldı)" if res["timed_out"] else "")


def job_full(inst):
    path = inst["dir"] / "concorde_full.json"
    if path.exists():
        return "önbellekte"
    n = inst["n"]
    wd = WORK_ROOT / "concorde_runs" / inst["name"] / "full"
    res = C.run_concorde(CONCORDE, inst["problem_file"], wd, ["-x", "-s", CONCORDE_SEED, "-o", "opt.sol"],
                         FULL_TIMEOUT_S)
    shutil.copy2(wd / "stdout.txt", OUTPUT_ROOT / "logs" / f"{inst['name']}_full.txt")
    rec = {k: v for k, v in res.items() if k != "workdir"}
    rec.update(tour_ok=False, known_opt=inst["known_opt"], matches_known=None)
    sol = wd / "opt.sol"
    if not res["timed_out"] and res["optimal"] is not None and sol.exists():
        tour = C.read_concorde_tour(sol, n)
        length = float(T.tour_length(inst["xy"], tour, inst["ew"]))
        rec.update(tour_length=length, tour_ok=bool(abs(length - res["optimal"]) < 0.5))
        if rec["tour_ok"]:
            save_npz(inst["dir"] / "opt_tour.npz", tour=tour.astype(np.int32))
        if inst["known_opt"] is not None:
            rec["matches_known"] = bool(abs(res["optimal"] - inst["known_opt"]) < 0.5)
    write_json(path, rec)
    shutil.rmtree(wd, ignore_errors=True)
    if rec["tour_ok"]:
        return f"kanıtlı optimum {res['optimal']:.0f} ({res['wall']:.0f} s, {res['bbnodes']} dal düğümü)"
    return f"optimum yok ({'süre aşıldı' if res['timed_out'] else 'hata'}, {res['wall']:.0f} s)"


# -------------------------------------------------------------------------------------- D: örnek başına analiz
def lists_of(inst, L, Mo, rank_alpha, rank_near, D):
    """Beşli listeler ve uzun sıralamalar. Sayma: BetaTour grafındaki kenarlar, altı ilk deneme turunda kullanılma
    sıklığına göre (eşitlikte kısa kenar önce); çalışmadaki sayma kontrolü gibi."""
    n = inst["n"]
    NB, rank_beta = Mo["NB"].astype(np.int64), Mo["rank"].astype(np.int64)
    succ = np.stack([T.tour_succ_pred(t)[0] for t in L["first_tours"]], axis=1)
    freq = T.ensemble_frequency(NB, succ)
    D_nb = D[np.arange(n)[:, None], NB]
    rank_count = np.take_along_axis(NB, np.lexsort((NB, D_nb, -freq), axis=1), 1)
    pop = L["pop_cand"][0].astype(np.int64)
    lists5 = {"nearest": rank_near[:, :5], "alpha_lkh": L["alpha_cand"].astype(np.int64),
              "alpha_ours": rank_alpha[:, :5], "popmusic": pop, "count": rank_count[:, :5],
              "betatour": rank_beta[:, :5], "count_mix": T.mix_candidates(rank_count, pop, 3, 5),
              "betatour_mix": T.mix_candidates(rank_beta, pop, 3, 5)}
    long = {"nearest": rank_near, "alpha_ours": rank_alpha, "count": rank_count, "betatour": rank_beta}
    return lists5, long, NB, freq


def analyse(inst):
    path = inst["dir"] / "analysis.json"
    if path.exists():
        return "önbellekte"
    n, xy, ew = inst["n"], inst["xy"], inst["ew"]
    L = np.load(inst["dir"] / "lkh.npz")
    Mo = np.load(inst["dir"] / "model.npz")
    root = read_json(inst["dir"] / "concorde_root.json") or {}
    full = read_json(inst["dir"] / "concorde_full.json") or {}
    opt = float(full["optimal"]) if full.get("tour_ok") else None
    opt_tour = np.load(inst["dir"] / "opt_tour.npz")["tour"].astype(np.int64) if opt is not None else None
    ub_lkh = float(L["ub_cost"])
    ubs = {"lkh": ub_lkh}
    if opt is not None:
        ubs["opt"] = opt
    t0 = time.time()
    D = C.distance_matrix(xy, ew)
    out = dict(name=inst["name"], group=inst["group"], n=n, opt=opt, known_opt=inst["known_opt"],
               matches_known=full.get("matches_known"), ub_lkh=ub_lkh,
               ub_gap_pct=None if opt is None else 100.0 * (ub_lkh - opt) / opt,
               concorde_full_s=full.get("wall"), concorde_full_timed_out=full.get("timed_out"),
               bbnodes=full.get("bbnodes"), concorde_root_s=root.get("wall"), lkh_times=L["times"].tolist(),
               model_s=float(Mo["t_prep"] + Mo["t_model"]))
    rem, unresolved = {}, {}
    # C1: 1-tree sınırı ve alfa değerleri, tam sayı aritmetiğiyle (LKH'nin düğüm cezalarıyla)
    tree = C.one_tree(D, L["pi"])
    W = tree["W"] / tree["precision"]
    out["c1"] = dict(bound=W, lkh_lower_bound=float(L["lkh_lb"]), gap_pct=None if opt is None else
                     100.0 * (opt - W) / opt)
    for u, ub in ubs.items():
        rem[("c1", u)] = C.c1_remaining(tree, ub)
    if opt is not None:
        out["c1"]["opt_edges_eliminated"] = int((~np.isin(C.tour_keys(opt_tour, n), rem[("c1", "opt")])).sum())
    rank_alpha = C.alpha_ranking(tree["alpha"], D, RANK_DEPTH)
    out["alpha_top5_agreement"] = float(np.mean([set(rank_alpha[v, :5].tolist()) == set(L["alpha_cand"][v].tolist())
                                                 for v in range(n)]))
    del tree
    out["c1"]["time_s"] = time.time() - t0
    # C2: Concorde'un kök LP'si (indirgenmiş maliyetler; denetimler kanıtlı optimuma karşı)
    rk = inst["dir"] / "rc_keep.npz"
    if root.get("dump") and root.get("lb") is not None and rk.exists():
        R = np.load(rk)
        lb = float(root["lb"])
        thr = None if root["dump"]["complete"] else R["thr"]
        quantum = float(root["dump"]["quantum"])
        tol = max(RC_TOL, quantum)
        c2 = dict(lower_bound=lb, gap_pct=None if opt is None else 100.0 * (opt - lb) / opt, dump=root["dump"],
                  tol=tol)
        for u, ub in ubs.items():
            r, un = C.c2_remaining(R["keys"], R["rc"], lb, ub, n, tol, thr=thr)
            rem[("c2", u)], unresolved[("c2", u)] = r, un
            c2[f"unresolved_share_{u}"] = 0.0 if un is None else float(un.mean())
        if opt is not None:
            c2["checks"] = C.c2_checks(R["keys"], R["rc"], lb, n, opt, opt_tour, tol, thr=thr,
                                       cap=root["dump"]["cap"], sum_negative=root["dump"]["sum_negative"],
                                       quantum=quantum)
        out["c2"] = c2
    elif root.get("lb") is not None:
        out["c2"] = dict(lower_bound=float(root["lb"]), gap_pct=None if opt is None else
                         100.0 * (opt - root["lb"]) / opt, dump=None)
    # Listeler: kanıtlı optimumun kaçırılan kenarları ve kanıtla tamlık
    rank_near = C.nearest_ranking(D, RANK_DEPTH)
    lists5, long, NB, freq = lists_of(inst, L, Mo, rank_alpha, rank_near, D)
    if opt is not None:
        out["missing_opt_pct"] = {name: 100.0 * T.candidate_coverage(lst, opt_tour, ks=[5])[5]["missing"] / n
                                  for name, lst in lists5.items()}
    out["certified"] = {}
    for (cert, u), r in rem.items():
        un = unresolved.get((cert, u))
        entry = dict(degree=C.degree_summary(r, n), unresolved_share=0.0 if un is None else float(un.mean()))
        entry["lists5"] = {name: dict(complete_share=float((C.certified_k(lst, r, n, un) <= 5).mean()),
                                      edge_coverage=C.edge_coverage(lst, r, n)) for name, lst in lists5.items()}
        entry["long"] = {name: C.k_summary(C.certified_k(lst, r, n, un), lst.shape[1]) for name, lst in long.items()}
        out["certified"][f"{cert}_{u}"] = entry
    # AUC: BetaTour grafındaki kenarlar arasında kalanları (kanıtla elenemeyen) ayırt etme
    for key in [("c2", "opt"), ("c2", "lkh"), ("c1", "opt"), ("c1", "lkh")]:
        if key in rem:
            break
    edges, prob = C.graph_edges(NB, Mo["prob"], n)
    _, fq = C.graph_edges(NB, freq, n)
    label = np.isin(edges, rem[key])
    dist = D[edges // n, edges % n].astype(np.float64)
    out["auc"] = dict(certificate=f"{key[0]}_{key[1]}", edges=int(len(edges)), remaining=int(label.sum()),
                      betatour=C.auc(prob, label), count=C.auc(fq, label), neg_distance=C.auc(-dist, label))
    out["analysis_s"] = time.time() - t0
    write_json(path, out)
    return f"analiz {out['analysis_s']:.0f} s"


def run_parallel(func, items, workers, label):
    """İşleri paralel yürütür; hatalar kaydedilir, diğer işler sürer."""
    errors = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(func, item): item for item in items}
        for fut in as_completed(futures):
            item = futures[fut]
            try:
                say(f"{label}: {item['name']}: {fut.result()}")
            except Exception as exc:
                errors.append((item["name"], repr(exc)))
                say(f"{label}: {item['name']}: HATA {exc!r}")
                with open(OUTPUT_ROOT / "logs" / "errors.txt", "a") as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {label} {item['name']}\n"
                            f"{traceback.format_exc()}\n")
    return errors
