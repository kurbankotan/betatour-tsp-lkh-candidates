def warmup():
    """Model ve LKH ısınması (ölçüme dahil değildir)."""
    xy = T.to_int_coords(T.gen_instance(100, "uniform", np.random.default_rng(9090)))
    succ = np.stack([np.roll(np.arange(100), -1)] * W, axis=1)
    M.predict_tourfold(MODEL, xy, K_GRAPH, DEVICE, ens_succ=succ, ens_cost=np.ones(W))
    first = A.task_first_trial(LKH, str(WORK_ROOT / "lkh_tmp"), SEEDS[0], xy, k=K_LIST, timeout=60)
    assert T.is_valid_tour(first["tour"], 100) and first["cand"].shape == (100, K_LIST) == first["alpha"].shape


def preflight():
    """Uzun koşudan önce, ölçüm dışı denetim: her TSPLIB örneğinde tek tohumla ilk deneme (dosya, mesafe türü, LKH'nin
    maliyeti, aday dosyasının biçimi) ve en büyük örnekte bir model tahmini (GPU belleği ve tek seferlik hazırlık)."""
    tsplib = [x for x in INSTANCES if x["group"] == "tsplib"]
    raw = parallel_tasks(A.task_first_trial, [dict(instance_common(x), seed=SEEDS[0], k=K_LIST, timeout=900)
                                              for x in tsplib])
    rows = {}
    for x, res in zip(tsplib, raw):
        n = len(x["xy"])
        assert len(res["events"]) == 1 and res["events"][0][1] == int(round(res["cost"])), \
            f"{x['name']}: LKH'nin yazdığı maliyet turun uzunluğuna eşit değil"
        present = res["cand"] >= 0
        rows[x["name"]] = dict(n=n, ew=x["ew"], budget_s=x["budget"], first_cost=int(res["cost"]),
                               candidates_per_node=float(present.sum() / n),
                               zero_alpha_per_node=float((present & (res["alpha"] == 0)).sum() / n),
                               nodes_with_father=int((res["dad"] >= 0).sum()), pre_time_s=float(res["pre_time"] or 0.0))
    biggest = max(INSTANCES, key=lambda x: len(x["xy"]))
    n = len(biggest["xy"])
    order = np.argsort(np.asarray(biggest["xy"])[:, 0], kind="stable")
    succ = np.empty(n, dtype=np.int32)
    succ[order] = np.roll(order, -1)
    start = time.perf_counter()
    M.predict_tourfold(MODEL, biggest["xy"], K_GRAPH, DEVICE, ens_succ=np.stack([succ] * W, axis=1), ens_cost=np.ones(W))
    return dict(tsplib=rows, largest_instance=biggest["name"], largest_nodes=n,
                first_model_call_s=float(time.perf_counter() - start))


warmup()

# Kodlama öz sınaması (run v21 ile aynı): bu makinede ve bu LKH ikilisiyle, sabit deneme sayılı koşular.
#   1) LKH'nin kendi alfa ve baba değerleriyle yazdığımız dosya, LKH'nin kendi dosyasıyla her denemede aynı maliyeti verir.
#   2) alfa = 0, 100, 200, ... dosyası aramayı en erken ikinci denemede değiştirir (yeniden başlatma kuralı).
SELFCHECK = A.encoding_selfcheck(LKH, str(WORK_ROOT / "lkh_tmp"),
                                 T.to_int_coords(T.gen_instance(2000, "uniform", np.random.default_rng(9092))),
                                 SEEDS[0], trials=25, k=K_LIST)
first_diff = SELFCHECK["first_trial_with_a_different_cost"]
assert SELFCHECK["native_file_equals_lkh_file"], "LKH'nin kendi alfa ve baba değerleriyle yazılan dosya LKH'nin dosyasıyla aynı aramayı vermedi."
assert SELFCHECK["tree_rule_matches_lkh"], "Ağaç kuralı LKH'nin alfa = 0 girdilerini üretmedi."
assert first_diff["rank_dad"] is not None and first_diff["rank_dad"] >= 2, \
    "alfa = 0, 100, 200, ... kodlaması aramayı beklenen biçimde değiştirmedi."
SELFCHECK["session_id"] = SESSION_ID
SELFCHECK["preflight"] = preflight()
write_json_atomic(OUTPUT_ROOT / "summary" / "selfcheck.json", SELFCHECK)
print("Öz sınama ve ön denetim tamam.")
for name, row in SELFCHECK["preflight"]["tsplib"].items():
    print(f"  {name:<12} n = {row['n']:>6}, {row['ew']}, bütçe {row['budget_s']:.0f} s, düğüm başına aday "
          f"{row['candidates_per_node']:.2f}, alfa = 0 olan {row['zero_alpha_per_node']:.3f}")
