def round_robin(lists):
    out = []
    for i in range(max(len(x) for x in lists)):
        out += [x[i] for x in lists if i < len(x)]
    return out


def make_instances():
    """Yeni sentetik örnekler. Sıra: boyuta göre, her boyutta aileler dönüşümlü (önce 10.000 düğüm)."""
    groups = {}
    for group, n, count, family, gen_seed, budget in SYNTHETIC:
        rng = np.random.default_rng(gen_seed)
        groups[group] = [dict(group=group, family=family, index=i, name=f"u27_{group}_{i:03d}",
                              xy=T.to_int_coords(T.gen_instance(n, family, rng)), ew="EUC_2D", problem_file=None,
                              budget=float(budget), optimum=None) for i in range(count)]
    result = []
    for size in sorted({n for _, n, *_ in SYNTHETIC}):
        result += round_robin([groups[g] for g, n, *_ in SYNTHETIC if n == size])
    assert len({x["name"] for x in result}) == len(result) == sum(len(v) for v in groups.values())
    return result


INSTANCES = make_instances()


def estimated_seconds(inst):
    """Örnek başına: kayıtlı LKH koşusu (bütçe), iki GA-EAX süreci, LKH'nin kendi birleştirmesi ve üç tam çözüm
    süreci (her biri bütçenin kalanı; tam çözüm süreçleri aynı anda koşar) ve küçük ek süreler."""
    n, budget = len(inst["xy"]), inst["budget"]
    return (budget + STOP_PAD_S) + 3.0 * (1.0 - SPLIT) * budget + 0.002 * n + 25.0


by_group = {}
for x in INSTANCES:
    by_group[x["group"]] = by_group.get(x["group"], 0.0) + estimated_seconds(x) / 3600.0
estimate_h = sum(by_group.values())
print(f"{len(INSTANCES)} örnek hazır. Tahmini süre: yaklaşık {estimate_h:.1f} saat ("
      + ", ".join(f"{g}: {h:.1f} sa" for g, h in by_group.items()) + ").")
