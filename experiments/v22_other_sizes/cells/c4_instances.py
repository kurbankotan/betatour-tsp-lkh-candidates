def tsplib_budget(n):
    return float(max(10.0, 0.02 * n))


def round_robin(lists):
    out = []
    for i in range(max(len(x) for x in lists)):
        out += [x[i] for x in lists if i < len(x)]
    return out


def make_instances():
    """Yeni sentetik örnekler (SYNTHETIC) ve önceki koşuların TSPLIB örnekleri. Sıra: önce küçük sentetik örnekler,
    sonra büyük sentetik örnekler ile TSPLIB örnekleri dönüşümlü (kısa oturumlar da dengeli olsun)."""
    groups = {}
    for group, n, count, family, gen_seed, budget in SYNTHETIC:
        rng = np.random.default_rng(gen_seed)
        groups[group] = [dict(group=group, family=family, index=i, name=f"s22_{group}_{i:03d}",
                              xy=T.to_int_coords(T.gen_instance(n, family, rng)), ew="EUC_2D", problem_file=None,
                              budget=float(budget), optimum=None) for i in range(count)]
    if TEST_MODE:                       # iki küçük TSPLIB biçimli dosya (biri CEIL_2D), ağ gerekmez
        folder = WORK_ROOT / "tsplib_test"
        folder.mkdir(exist_ok=True)
        specs = {"test_euc800": (800, "uniform", "EUC_2D", 71), "test_ceil700": (700, "clustered", "CEIL_2D", 72)}
        paths = {}
        for name in TSPLIB_NAMES:
            n, family, ew, seed = specs[name]
            paths[name] = str(folder / (name + ".tsp"))
            T.write_tsp(paths[name], T.to_int_coords(T.gen_instance(n, family, np.random.default_rng(seed))),
                        name=name, ew=ew)
        optimum = {name: None for name in TSPLIB_NAMES}
    else:
        paths = T.download_tsplib(str(WORK_ROOT / "tsplib"), names=TSPLIB_NAMES)
        absent = sorted(set(TSPLIB_NAMES) - set(paths))
        assert not absent, f"Eksik TSPLIB dosyaları: {absent}. Ağ bağlantısını denetleyip bu hücreyi yeniden çalıştırın."
        optimum = {name: float(T.TSPLIB_OPT[name]) for name in TSPLIB_NAMES}
    groups["tsplib"] = []
    for i, name in enumerate(TSPLIB_NAMES):
        data = T.read_tsplib(paths[name])
        n, ew = int(data["dimension"]), data["edge_weight_type"]
        assert len(data["xy"]) == n and ew in {"EUC_2D", "CEIL_2D"}, (name, n, ew)
        groups["tsplib"].append(dict(group="tsplib", family="tsplib", index=i, name=name, xy=data["xy"], ew=ew,
                                     problem_file=paths[name], budget=6.0 if TEST_MODE else tsplib_budget(n),
                                     optimum=optimum[name]))
    small = round_robin([groups[g] for g, n, *_ in SYNTHETIC if n <= 2000] or [[]])
    large = round_robin([groups[g] for g, n, *_ in SYNTHETIC if n > 2000] + [groups["tsplib"]])
    result = small + large
    assert len({x["name"] for x in result}) == len(result) == sum(len(v) for v in groups.values())
    return result


INSTANCES = make_instances()


def estimated_seconds(inst):
    phase = 0.0016 * len(inst["xy"])            # ortak ilk aşama: run v5 ve v21'de ölçülen sürelere göre
    return phase + inst["budget"] + PAD_SECONDS + (len(ARM_NAMES) - 1) * (inst["budget"] - phase + PAD_SECONDS + 2.0)


estimate_h = sum(estimated_seconds(x) for x in INSTANCES) / 3600.0
by_group = {}
for x in INSTANCES:
    by_group[x["group"]] = by_group.get(x["group"], 0.0) + estimated_seconds(x) / 3600.0
print(f"{len(INSTANCES)} örnek hazır; {len(ARM_NAMES)} kol için tahmini toplam süre yaklaşık {estimate_h:.1f} saat ("
      + ", ".join(f"{g}: {h:.1f} sa" for g, h in by_group.items()) + ").")
