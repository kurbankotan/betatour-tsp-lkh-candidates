def warmup():
    """Model ve LKH ısınması (ölçüme dahil değildir)."""
    xy = T.to_int_coords(T.gen_instance(100, "uniform", np.random.default_rng(9090)))
    succ = np.stack([np.roll(np.arange(100), -1)] * W, axis=1)
    M.predict_tourfold(MODEL, xy, K_GRAPH, DEVICE, ens_succ=succ, ens_cost=np.ones(W))
    first = A.task_first_trial(LKH, str(WORK_ROOT / "lkh_tmp"), SEEDS[0], xy, k=K_LIST, timeout=60)
    assert T.is_valid_tour(first["tour"], 100) and first["cand"].shape == (100, K_LIST) == first["alpha"].shape
    # Deney boyutunda bir tahmin: ilk örnek, GPU'nun tek seferlik hazırlık süresini ödemesin.
    big = T.to_int_coords(T.gen_instance(N_NODES, "uniform", np.random.default_rng(9091)))
    order = np.argsort(big[:, 0], kind="stable")
    succ_big = np.empty(N_NODES, dtype=np.int32)
    succ_big[order] = np.roll(order, -1)
    M.predict_tourfold(MODEL, big, K_GRAPH, DEVICE, ens_succ=np.stack([succ_big] * W, axis=1), ens_cost=np.ones(W))


warmup()

# Kodlama öz sınaması: bu makinede ve bu LKH ikilisiyle, sabit deneme sayılı (belirlenimci) koşular. LKH'nin kendi
# yazdığı aday dosyasıyla yapılan arama, yeniden yazılmış dört dosyayla yapılan aramalarla karşılaştırılır:
#   1) LKH'nin kendi alfa değerleri + baba alanı: her denemede aynı maliyet (dosyamız LKH'ninkiyle eşdeğer).
#   2) alfa = 0, 100, 200, ... + baba alanı: arama en erken ikinci denemede ayrılır (yeniden başlatma kuralı).
#   3) Baba alanı olmayan dosyalar (run v1-v8 böyle yazıldı): fark ilk denemede de başlayabilir.
# Ayrıca ağaç kuralının LKH'nin alfa = 0 girdilerini aynen ürettiği denetlenir.
SELFCHECK = A.encoding_selfcheck(LKH, str(WORK_ROOT / "lkh_tmp"),
                                 T.to_int_coords(T.gen_instance(2000, "uniform", np.random.default_rng(9092))),
                                 SEEDS[0], trials=25, k=K_LIST)
SELFCHECK["session_id"] = SESSION_ID
write_json_atomic(OUTPUT_ROOT / "summary" / "selfcheck.json", SELFCHECK)
first_diff = SELFCHECK["first_trial_with_a_different_cost"]
assert SELFCHECK["native_file_equals_lkh_file"], "LKH'nin kendi alfa ve baba değerleriyle yazılan dosya LKH'nin dosyasıyla aynı aramayı vermedi."
assert SELFCHECK["tree_rule_matches_lkh"], "Ağaç kuralı LKH'nin alfa = 0 girdilerini üretmedi."
assert first_diff["rank_dad"] is not None and first_diff["rank_dad"] >= 2, \
    "alfa = 0, 100, 200, ... kodlaması aramayı beklenen biçimde değiştirmedi; deneyin öncülü bu LKH ikilisinde geçerli değil."
print("Öz sınama tamam. Yeniden başlatmada korunabilen tur kenarı payı: LKH'nin kendi alfası "
      f"{SELFCHECK['kept_share_native']:.3f}, rank kodlaması {SELFCHECK['kept_share_rank']:.3f}. "
      f"Aramanın LKH'nin kendi dosyasından ayrıldığı ilk deneme: {first_diff}")
