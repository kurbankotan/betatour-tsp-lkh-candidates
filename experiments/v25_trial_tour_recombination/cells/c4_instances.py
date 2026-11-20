def tsplib_budget(n):
    return float(max(10.0, 0.02 * n))


def round_robin(lists):
    out = []
    for i in range(max(len(x) for x in lists)):
        out += [x[i] for x in lists if i < len(x)]
    return out


def make_instances():
    """Yeni sentetik örnekler ve optimumu bilinen TSPLIB örnekleri. Sıra: 10.000 düğümlü örnekler (aileler
    dönüşümlü), TSPLIB örnekleri, en son en büyük örnekler; kısa bir oturum da ilk soruyu yanıtlar."""
    groups = {}
    for group, n, count, family, gen_seed, budget in SYNTHETIC:
        rng = np.random.default_rng(gen_seed)
        groups[group] = [dict(group=group, family=family, index=i, name=f"p25_{group}_{i:03d}",
                              xy=T.to_int_coords(T.gen_instance(n, family, rng)), ew="EUC_2D", problem_file=None,
                              budget=float(budget), optimum=None) for i in range(count)]
    if TEST_MODE:                       # TSPLIB biçimli küçük bir dosya, ondalıklı koordinatlarla; ağ gerekmez
        folder = WORK_ROOT / "tsplib_test"
        folder.mkdir(exist_ok=True)
        paths, optimum = {}, {}
        for name in TSPLIB_NAMES:
            xy = np.random.default_rng(73).random((900, 2)) * 5000.0
            paths[name] = str(folder / (name + ".tsp"))
            with open(paths[name], "w") as f:
                f.write(f"NAME : {name}\nTYPE : TSP\nDIMENSION : 900\nEDGE_WEIGHT_TYPE : EUC_2D\nNODE_COORD_SECTION\n")
                f.writelines(f"{i} {x:.3f} {y:.3f}\n" for i, (x, y) in enumerate(xy, 1))
                f.write("EOF\n")
            optimum[name] = None
    else:
        paths = T.download_tsplib(str(WORK_ROOT / "tsplib"), names=TSPLIB_NAMES)
        absent = sorted(set(TSPLIB_NAMES) - set(paths))
        assert not absent, f"Eksik TSPLIB dosyaları: {absent}. Ağ bağlantısını denetleyip bu hücreyi yeniden çalıştırın."
        optimum = {name: float(T.TSPLIB_OPT[name]) for name in TSPLIB_NAMES}
    groups["tsplib"] = []
    for i, name in enumerate(TSPLIB_NAMES):
        data = T.read_tsplib(paths[name])
        n, ew = int(data["dimension"]), data["edge_weight_type"]
        assert len(data["xy"]) == n and ew == "EUC_2D", (name, n, ew)       # GA-EAX yalnızca EUC_2D okur
        groups["tsplib"].append(dict(group="tsplib", family="tsplib", index=i, name=name, xy=data["xy"], ew=ew,
                                     problem_file=paths[name], budget=10.0 if TEST_MODE else tsplib_budget(n),
                                     optimum=optimum[name]))
    synthetic = [(g, n) for g, n, *_ in SYNTHETIC]
    smallest = min(n for _, n in synthetic)
    first = round_robin([groups[g] for g, n in synthetic if n == smallest])
    last = round_robin([groups[g] for g, n in synthetic if n != smallest] or [[]])
    result = first + groups["tsplib"] + last
    assert len({x["name"] for x in result}) == len(result) == sum(len(v) for v in groups.values())
    return result


INSTANCES = make_instances()


def estimated_seconds(inst):
    """Örnek başına: kayıtlı LKH koşusu (bütçe + ön işleme), GA-EAX birleştirmeleri (en uzunu bütçenin
    1 - en küçük bölme kadarı), bilinen melez (bütçe) ve küçük ek süreler."""
    n, budget = len(inst["xy"]), inst["budget"]
    return (budget + STOP_PAD_S) + (1.0 - min(SPLITS)) * budget + budget + 0.0015 * n + 20.0


def reference_seconds():
    """Sıfırdan GA-EAX: altışar örnek paralel koşar; bir grubun süresi 6 x en uzun bütçedir."""
    pending = [x for x in INSTANCES if len(x["xy"]) <= REFERENCE_MAX_N]
    return sum(W * max(x["budget"] for x in pending[i:i + W]) + 30.0 for i in range(0, len(pending), W))


by_group = {}
for x in INSTANCES:
    by_group[x["group"]] = by_group.get(x["group"], 0.0) + estimated_seconds(x) / 3600.0
estimate_h = sum(by_group.values())
reference_h = reference_seconds() / 3600.0 if RUN_REFERENCE else 0.0
print(f"{len(INSTANCES)} örnek hazır. Tahmini süre: ana koşu yaklaşık {estimate_h:.1f} saat ("
      + ", ".join(f"{g}: {h:.1f} sa" for g, h in by_group.items())
      + f"), sıfırdan GA-EAX referansı yaklaşık {reference_h:.1f} saat; toplam yaklaşık {estimate_h + reference_h:.1f} saat.")
