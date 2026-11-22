def tsplib_budget(n):
    return float(max(10.0, 0.02 * n))


def round_robin(lists):
    out = []
    for i in range(max(len(x) for x in lists)):
        out += [x[i] for x in lists if i < len(x)]
    return out


def make_instances():
    """Yeni sentetik örnekler ve optimumu bilinen TSPLIB örnekleri. Sıra: sentetik örnekler boyuta göre (önce 20.000
    düğüm; her boyutta aileler dönüşümlü), sonra TSPLIB örnekleri."""
    groups = {}
    for group, n, count, family, gen_seed, budget in SYNTHETIC:
        rng = np.random.default_rng(gen_seed)
        groups[group] = [dict(group=group, family=family, index=i, name=f"s29_{group}_{i:03d}",
                              xy=T.to_int_coords(T.gen_instance(n, family, rng)), ew="EUC_2D", problem_file=None,
                              budget=float(budget), optimum=None) for i in range(count)]
    names = [name for _, members in TSPLIB_SETS for name in members]
    if TEST_MODE:                       # TSPLIB biçimli küçük dosya, ondalıklı koordinatlarla; ağ gerekmez
        folder = WORK_ROOT / "tsplib_test"
        folder.mkdir(exist_ok=True)
        paths, optimum = {}, {}
        for name in names:
            count = int(name[len("test_euc"):])
            xy = np.random.default_rng(73 + count).random((count, 2)) * 5000.0
            paths[name] = str(folder / (name + ".tsp"))
            with open(paths[name], "w") as f:
                f.write(f"NAME : {name}\nTYPE : TSP\nDIMENSION : {count}\nEDGE_WEIGHT_TYPE : EUC_2D\nNODE_COORD_SECTION\n")
                f.writelines(f"{i} {x:.3f} {y:.3f}\n" for i, (x, y) in enumerate(xy, 1))
                f.write("EOF\n")
            optimum[name] = None
    else:
        paths = T.download_tsplib(str(WORK_ROOT / "tsplib"), names=names)
        absent = sorted(set(names) - set(paths))
        assert not absent, f"Eksik TSPLIB dosyaları: {absent}. Ağ bağlantısını denetleyip bu hücreyi yeniden çalıştırın."
        optimum = {name: float(T.TSPLIB_OPT[name]) for name in names}
    for group, members in TSPLIB_SETS:
        groups[group] = []
        for i, name in enumerate(members):
            data = T.read_tsplib(paths[name])
            n, ew = int(data["dimension"]), data["edge_weight_type"]
            assert len(data["xy"]) == n and ew == "EUC_2D", (name, n, ew)       # GA-EAX yalnızca EUC_2D okur
            groups[group].append(dict(group=group, family="tsplib", index=i, name=name, xy=data["xy"], ew=ew,
                                      problem_file=paths[name], budget=10.0 if TEST_MODE else tsplib_budget(n),
                                      optimum=optimum[name]))
    result = []
    for size in sorted({n for _, n, *_ in SYNTHETIC}):
        result += round_robin([groups[g] for g, n, *_ in SYNTHETIC if n == size])
    for group, _ in TSPLIB_SETS:
        result += groups[group]
    assert len({x["name"] for x in result}) == len(result) == sum(len(v) for v in groups.values())
    return result


INSTANCES = make_instances()


def estimated_seconds(inst):
    """Örnek başına: kayıtlı LKH koşusu (bütçe), iki GA-EAX süreci, LKH'nin kendi birleştirmesi ve kademeli
    birleştirme (her biri bütçenin kalanı) ve kayıtların okunması gibi küçük ek süreler."""
    n, budget = len(inst["xy"]), inst["budget"]
    return (budget + STOP_PAD_S) + 3.0 * (1.0 - SPLIT) * budget + 0.003 * n + 25.0


by_group = {}
for x in INSTANCES:
    by_group[x["group"]] = by_group.get(x["group"], 0.0) + estimated_seconds(x) / 3600.0
estimate_h = sum(by_group.values())
print(f"{len(INSTANCES)} örnek hazır. Tahmini süre: yaklaşık {estimate_h:.1f} saat ("
      + ", ".join(f"{g}: {h:.1f} sa" for g, h in by_group.items()) + ").")
