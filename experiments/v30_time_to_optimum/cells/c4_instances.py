# Örnekler (1 dk): pilotun 18 sentetik örneği (yeniden üretilir, dosyaların SHA-256 değeri denetlenir; optimumlar
# pilotta Concorde ile kanıtlandı) ve yayımlanmış optimumu olan 19 TSPLIB örneği. Pilotun klasörü Drive'da varsa
# kaydettiği optimal turlar da okunur (yalnızca kapsama analizi için; uzunlukları denetlenir).
def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_instances():
    problems = WORK_ROOT / "problems"
    problems.mkdir(exist_ok=True)
    out = []
    for group, n, count, family, seed, keep in SYNTHETIC:
        rng = np.random.default_rng(seed)
        for i in range(count):
            xy = T.to_int_coords(T.gen_instance(n, family, rng))
            if i not in keep:
                continue
            name = f"cp1_{group}_{i:02d}"
            pf = problems / f"{name}.tsp"
            T.write_tsp(pf, xy, name=name, ew="EUC_2D")
            assert sha256_of(pf) == PROBLEM_SHA256[name], f"{name}: örnek dosyası pilottakiyle aynı değil"
            out.append(dict(name=name, group=group, family=family, n=n, xy=xy, ew="EUC_2D", problem_file=str(pf),
                            optimum=int(OPTIMA[name]), optimum_source="Concorde (certified pilot)"))
    if TEST_MODE:
        paths = dict(_test["tsplib_files"])
    else:
        paths = T.download_tsplib(str(WORK_ROOT / "tsplib"), names=TSPLIB_NAMES)
        absent = sorted(set(TSPLIB_NAMES) - set(paths))
        assert not absent, f"Eksik TSPLIB dosyaları: {absent}. Ağ bağlantısını denetleyip bu hücreyi yeniden çalıştırın."
    for name in TSPLIB_NAMES:
        data = T.read_tsplib(paths[name])
        n, ew = int(data["dimension"]), data["edge_weight_type"]
        assert len(data["xy"]) == n and ew == "EUC_2D", (name, n, ew)
        pf = problems / f"{name}.tsp"
        shutil.copy2(paths[name], pf)
        optimum = TSPLIB_TEST_OPTIMA[name] if TEST_MODE else T.TSPLIB_OPT[name]
        out.append(dict(name=name, group="tsplib", family="tsplib", n=n, xy=data["xy"], ew=ew, problem_file=str(pf),
                        optimum=int(optimum), optimum_source="test (exact MILP)" if TEST_MODE else "TSPLIB"))
    for inst in out:
        inst["cap"] = CAP_PER_NODE * inst["n"]
        inst["dir"] = OUTPUT_ROOT / "instances" / inst["name"]
        inst["dir"].mkdir(parents=True, exist_ok=True)
        inst["sha256"] = sha256_of(inst["problem_file"])
        inst["opt_tour"] = None
        pilot = Path(PILOT_DIR) / "instances" / inst["name"] / "opt_tour.npz" if PILOT_DIR else None
        if pilot is not None and pilot.exists():
            tour = np.load(pilot)["tour"].astype(np.int64)
            if T.is_valid_tour(tour, inst["n"]) and T.tour_length(inst["xy"], tour, inst["ew"]) == inst["optimum"]:
                inst["opt_tour"] = tour
            else:
                print(f"UYARI: {inst['name']}: pilotun optimal turu bu örneğe uymuyor; kapsama analizinde kullanılmaz.")
    assert len({x["name"] for x in out}) == len(out)
    return out


INSTANCES = build_instances()
BY_NAME = {inst["name"]: inst for inst in INSTANCES}
print(f"{len(INSTANCES)} örnek: " + ", ".join(f"{g} {sum(i['group'] == g for i in INSTANCES)}"
                                              for g in dict.fromkeys(i["group"] for i in INSTANCES))
      + f" | pilotun optimal turu olan: {sum(i['opt_tour'] is not None for i in INSTANCES)} | toplam üst sınır "
      f"{sum(i['cap'] for i in INSTANCES) / 60:.0f} dk/kol/tohum")
