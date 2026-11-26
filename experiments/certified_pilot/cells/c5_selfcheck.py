# Öz sınama (1-2 dk): Concorde'un yardım metni ve -z dökümünün anlamı; C1 ve C2 denetimleri küçük bir örnekte.
# C2 yalnızca şu durumda kullanılır: -z dökümü bütün grafı kapsıyor (z = n-1) ve kanıtlı optimuma karşı denetimleri
# geçiyor. Düğüm başına kısmi döküm (z = K_RC) ayrıca doğrulanırsa deneyde o kullanılır (dosyalar küçük kalır).
import re

SELF = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "n": SELF_N}
usage = C.concorde_usage(CONCORDE)
(OUTPUT_ROOT / "logs" / "concorde_usage.txt").write_text(usage)
SELF["usage_lists_z"] = bool(re.search(r"^\s*-z\b", usage, re.M))
sdir = OUTPUT_ROOT / "selfcheck"
sdir.mkdir(exist_ok=True)
for stale in ["lkh.npz", "model.npz", "concorde_full.json", "opt_tour.npz"]:
    (sdir / stale).unlink(missing_ok=True)
(WORK_ROOT / "problems").mkdir(exist_ok=True)
pf_s = WORK_ROOT / "problems" / "selfcheck.tsp"
xy_s = T.to_int_coords(T.gen_instance(SELF_N, "uniform", np.random.default_rng(424242)))
T.write_tsp(pf_s, xy_s, name="selfcheck")
SI = dict(name="selfcheck", group="selfcheck", family="uniform", n=SELF_N, xy=xy_s, ew="EUC_2D", known_opt=None,
          source=None, dir=sdir, problem_file=str(pf_s))
say("öz sınama:", stage_lkh(SI))
say("öz sınama:", job_full(SI))
say("öz sınama:", stage_model(SI))
full_s = read_json(sdir / "concorde_full.json")
assert full_s["tour_ok"], "Concorde öz sınama örneğini çözemedi; günlük: logs/selfcheck_full.txt"
opt_s = float(full_s["optimal"])
tour_s = np.load(sdir / "opt_tour.npz")["tour"].astype(np.int64)
L_s = np.load(sdir / "lkh.npz")
assert opt_s <= float(L_s["ub_cost"]) + 0.5, "Concorde'un optimumu LKH turundan uzun"

# C1: kendi 1-ağaç hesabımız (tam sayı) LKH'nin cezalarıyla; optimal turun hiçbir kenarı elenmemeli.
D_s = C.distance_matrix(xy_s)
tree_s = C.one_tree(D_s, L_s["pi"])
rank_s = C.alpha_ranking(tree_s["alpha"], D_s, 5)
SELF["c1"] = dict(bound=tree_s["W"] / 100.0, lkh_lower_bound=float(L_s["lkh_lb"]), opt=opt_s,
                  opt_edges_eliminated=int((~np.isin(C.tour_keys(tour_s, SELF_N), C.c1_remaining(tree_s, opt_s))).sum()),
                  alpha_top5_agreement=float(np.mean([set(rank_s[v].tolist()) == set(L_s["alpha_cand"][v].tolist())
                                                      for v in range(SELF_N)])))
assert SELF["c1"]["opt_edges_eliminated"] == 0 and SELF["c1"]["bound"] <= opt_s + 1e-9, SELF["c1"]
del tree_s

# C2: -z dökümünün anlamı. Önce z = n-1 (bütün kenarlar); kapsamıyorsa tam kenar dosyasıyla (-E) yeniden.
attempts, full_e_ok, dump_all = [], None, None
for full_e in (False, True):
    res, dump = concorde_root(SI, SELF_N - 1, full_e, "self_all" + ("_E" if full_e else ""))
    a = dict(full_e=full_e, z=SELF_N - 1, lb=res["lb"], returncode=res["returncode"], wall=res["wall"],
             dump=dump is not None)
    if dump is not None:
        complete = bool(len(dump["keys"]) == SELF_N * (SELF_N - 1) // 2)
        chk = C.c2_checks(dump["keys"], dump["rc"], res["lb"], SELF_N, opt_s, tour_s, max(RC_TOL, dump["quantum"]),
                          sum_negative=float(np.clip(-dump["rc"], 0, None).sum()), quantum=dump["quantum"])
        a.update(lines=dump["lines"], edges=int(len(dump["keys"])), complete=complete,
                 min_count=int(dump["counts"].min()), dup_max_diff=dump["dup_max_diff"], quantum=dump["quantum"],
                 checks=chk)
        if complete and chk["passed"]:
            full_e_ok, dump_all = full_e, dump
    attempts.append(a)
    if full_e_ok is not None:
        break
C2_MODE = None
if full_e_ok is not None:
    res, dump = concorde_root(SI, K_RC_SELF, full_e_ok, "self_partial")
    a = dict(full_e=full_e_ok, z=K_RC_SELF, lb=res["lb"], returncode=res["returncode"], wall=res["wall"],
             dump=dump is not None)
    partial_ok = False
    if dump is not None:
        tol_s = max(RC_TOL, dump["quantum"])
        thr = C.node_thresholds(dump["keys"], dump["rc"], SELF_N, K_RC_SELF)
        chk = C.c2_checks(dump["keys"], dump["rc"], res["lb"], SELF_N, opt_s, tour_s, tol_s, thr=thr,
                          sum_negative=float(np.clip(-dump["rc"], 0, None).sum()), quantum=dump["quantum"])
        listed = np.isin(dump_all["keys"], dump["keys"])
        ua, ub_ = dump_all["keys"][~listed] // SELF_N, dump_all["keys"][~listed] % SELF_N
        thresholds_valid = bool((dump_all["rc"][~listed] >= np.maximum(thr[ua], thr[ub_]) - tol_s).all())
        pos, found = C.lookup(dump_all["keys"], dump["keys"])
        same_rc = float(np.abs(dump_all["rc"][pos[found]] - dump["rc"][found]).max()) if found.any() else None
        per_node = bool(dump["counts"].min() >= K_RC_SELF)
        a.update(lines=dump["lines"], edges=int(len(dump["keys"])), min_count=int(dump["counts"].min()),
                 complete=bool(len(dump["keys"]) == SELF_N * (SELF_N - 1) // 2), quantum=dump["quantum"],
                 per_node=per_node, thresholds_valid=thresholds_valid, max_rc_difference_to_full_dump=same_rc,
                 lb_difference_to_full_dump=float(res["lb"] - attempts[-1]["lb"]), checks=chk)
        partial_ok = per_node and thresholds_valid and chk["passed"] and same_rc is not None and same_rc <= tol_s
    attempts.append(a)
    C2_MODE = dict(full_e=bool(full_e_ok), partial=bool(partial_ok))
SELF["c2_attempts"] = attempts
SELF["c2_mode"] = C2_MODE
write_json(sdir / "selfcheck.json", SELF)

print(f"Öz sınama ({SELF_N} düğüm): Concorde optimumu {opt_s:.0f}, LKH turu {float(L_s['ub_cost']):.0f}; "
      f"-z yardım metninde: {'evet' if SELF['usage_lists_z'] else 'HAYIR'}")
print(f"  C1: 1-ağaç sınırı {SELF['c1']['bound']:.2f} (LKH'nin yazdığı {SELF['c1']['lkh_lower_bound']:.2f}); optimal "
      f"turun elenen kenarı: {SELF['c1']['opt_edges_eliminated']}; alfa ilk-5 listeleri LKH ile aynı: "
      f"%{100 * SELF['c1']['alpha_top5_agreement']:.1f}")
for a in attempts:
    chk = a.get("checks") or {}
    print(f"  C2 denemesi z={a['z']}{' -E' if a['full_e'] else ''}: döküm {'var' if a['dump'] else 'YOK'}"
          + (f", {a['edges']} kenar (tam: {a.get('complete')}), düğüm başına en az {a['min_count']}, "
             f"kök farkı %{chk.get('root_gap_pct', float('nan')):.4f}, denetimler: "
             f"{'geçti' if chk.get('passed') else 'GEÇMEDİ'} (ikili gevşeklik {chk.get('dual_slack', float('nan')):.3f}, "
             f"elenen optimal kenar {chk.get('opt_edges_eliminated')})" if a["dump"] else ""))
if C2_MODE is None:
    print("UYARI: C2 kullanılamıyor. Pilot C1 ve kanıtlı optimum kapsamıyla sürer; G1-G3 değerlendirilemez. "
          "Lütfen logs/concorde_usage.txt ve selfcheck/selfcheck.json dosyalarını gönderin.")
else:
    print(f"C2 kipi: {'kısmi döküm (z = K_RC, eşiklerle)' if C2_MODE['partial'] else 'tam döküm (z = n-1)'}"
          f"{', tam kenar dosyasıyla (-E)' if C2_MODE['full_e'] else ''}")
