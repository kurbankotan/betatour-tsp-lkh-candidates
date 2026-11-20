def make_instances():
    """20 yeni sentetik EUC_2D örneği (20.000 düğüm): 10 uniform ve 10 kümeli; aileler dönüşümlü sıralanır."""
    result = []
    for family, group in (("uniform", "uni20k"), ("clustered", "clu20k")):
        rng = np.random.default_rng(GENERATOR_SEEDS[family])
        for i in range(PER_FAMILY):
            xy = T.to_int_coords(T.gen_instance(N_NODES, family, rng))
            result.append(dict(group=group, family=family, index=i, name=f"long_{group}_{i:03d}", xy=xy,
                               ew="EUC_2D", problem_file=None, budget=BUDGET_S))
    result.sort(key=lambda x: (x["index"], x["group"]))
    assert len({x["name"] for x in result}) == len(result) == 2 * PER_FAMILY
    return result


INSTANCES = make_instances()
# Run v22'de 20.000 düğümlü örneklerde ölçülen süreler: ortak ilk aşama 30-33 s; LKH'nin saati duvar saatinden
# yaklaşık %1,5 geri kaldığı için yeniden başlatılan kol kendi sınırından biraz uzun sürer.
# Kesintisiz kol kendi sınırında biter: LKH'nin sınırı ön işlemeyi (20.000 düğümde yaklaşık 15 s) saymaz.
phase_s, pre_s = (33.0, 15.0) if N_NODES >= 20000 else (2.0, 1.0)
full_s = min(BUDGET_S + PAD_SECONDS, 1.015 * (BUDGET_S + pre_s) + 2.0)
estimate_h = len(INSTANCES) * (phase_s + full_s + (len(ARM_NAMES) - 1) * (1.015 * (BUDGET_S - phase_s) + 4.0)) / 3600.0
print(f"{len(INSTANCES)} örnek hazır; {len(ARM_NAMES)} kol için tahmini toplam süre yaklaşık {estimate_h:.1f} saat "
      f"(örnek başına yaklaşık {estimate_h * 60 / len(INSTANCES):.1f} dk).")
