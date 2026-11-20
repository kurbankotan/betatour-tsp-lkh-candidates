def make_instances():
    """12 yeni sentetik EUC_2D örneği (10.000 düğüm): 6 uniform ve 6 kümeli; aileler dönüşümlü sıralanır."""
    result = []
    for family, group in (("uniform", "uni10k"), ("clustered", "clu10k")):
        rng = np.random.default_rng(GENERATOR_SEEDS[family])
        for i in range(PER_FAMILY):
            xy = T.to_int_coords(T.gen_instance(N_NODES, family, rng))
            result.append(dict(group=group, family=family, index=i, name=f"mark_{group}_{i:03d}", xy=xy,
                               ew="EUC_2D", problem_file=None, budget=BUDGET_S))
    result.sort(key=lambda x: (x["index"], x["group"]))
    assert len({x["name"] for x in result}) == len(result) == 2 * PER_FAMILY
    return result


INSTANCES = make_instances()
# Run v5'te 10.000 düğümlü örneklerde ölçülen süreler: ortak ilk aşama 12-14 s, yeniden başlatılan kol 108-111 s.
phase_s = 15.0 if N_NODES >= 10000 else 2.0
estimate_h = len(INSTANCES) * (phase_s + BUDGET_S + PAD_SECONDS
                               + (len(ARM_NAMES) - 1) * (BUDGET_S - phase_s + PAD_SECONDS + 2.0)) / 3600.0
print(f"{len(INSTANCES)} örnek hazır; {len(ARM_NAMES)} kol için tahmini toplam süre yaklaşık {estimate_h:.1f} saat "
      f"(örnek başına yaklaşık {estimate_h * 60 / len(INSTANCES):.1f} dk).")
