# Özet ve karar (birkaç saniye). Tablolar ekrana yazılır ve summary/ klasörüne kaydedilir (CSV ve JSON).
ROWS = [r for r in (read_json(i["dir"] / "analysis.json") for i in INSTANCES) if r]
SYN = [g for g, *_ in SYNTHETIC]
GROUPS = SYN + (["tsplib"] if any(r["group"] == "tsplib" for r in ROWS) else [])
BY = {g: [r for r in ROWS if r["group"] == g] for g in GROUPS}
LISTS = [("nearest", "en yakın"), ("alpha_lkh", "alfa-5 (LKH)"), ("alpha_ours", "alfa-5 (C1)"),
         ("popmusic", "POPMUSIC-5"), ("count", "sayma-5"), ("betatour", "BetaTour-5"), ("count_mix", "sayma 3+POP"),
         ("betatour_mix", "BetaTour 3+POP")]
LONG = [("nearest", "en yakın"), ("alpha_ours", "alfa (C1)"), ("count", "sayma"), ("betatour", "BetaTour")]
CERTS = [("c1_opt", "C1, ÜS=opt"), ("c1_lkh", "C1, ÜS=LKH"), ("c2_opt", "C2, ÜS=opt"), ("c2_lkh", "C2, ÜS=LKH")]
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def get(r, *path):
    for p in path:
        if r is None:
            return None
        r = r.get(p) if isinstance(r, dict) else None
    return r


def mean_of(rs, *path, scale=1.0):
    vals = [get(r, *path) for r in rs]
    vals = [float(v) for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return scale * float(np.mean(vals)) if vals else np.nan


def table(index, columns):
    frame = pd.DataFrame(columns, index=index)
    return frame


def show(title, frame, digits=3):
    print("\n" + title)
    print(frame.round(digits).to_string())


TABLES = {}
TABLES["1_optimum_ve_sinirlar"] = table(GROUPS, {
    "örnek": [len(BY[g]) for g in GROUPS],
    "kanıtlı opt": [sum(r["opt"] is not None for r in BY[g]) for g in GROUPS],
    "Concorde tam (s, ort)": [mean_of(BY[g], "concorde_full_s") for g in GROUPS],
    "Concorde tam (s, en çok)": [max([r["concorde_full_s"] or 0 for r in BY[g]] or [np.nan]) for g in GROUPS],
    "dal düğümü (ort)": [mean_of(BY[g], "bbnodes") for g in GROUPS],
    "kök LP (s, ort)": [mean_of(BY[g], "concorde_root_s") for g in GROUPS],
    "ÜS(LKH)-opt %": [mean_of(BY[g], "ub_gap_pct") for g in GROUPS],
    "opt-C2 kök %": [mean_of(BY[g], "c2", "gap_pct") for g in GROUPS],
    "opt-C1 1-ağaç %": [mean_of(BY[g], "c1", "gap_pct") for g in GROUPS],
    "TSPLIB opt aynı": [f"{sum(bool(r.get('matches_known')) for r in BY[g])}/{sum(r['known_opt'] is not None for r in BY[g])}"
                        for g in GROUPS]})
show("Tablo 1. Kanıtlı optimum ve alt sınırlar (farklar optimuma göre %)", TABLES["1_optimum_ve_sinirlar"], 4)

TABLES["2_kacirilan_opt_kenarlari"] = table(GROUPS, {label: [mean_of(BY[g], "missing_opt_pct", name) for g in GROUPS]
                                                     for name, label in LISTS})
show("Tablo 2. Kanıtlı optimal turun 5'li listelerde bulunmayan kenarları (tur kenarlarının %'si)",
     TABLES["2_kacirilan_opt_kenarlari"])

deg = {}
for cert, label in CERTS:
    deg[f"{label}: ort derece"] = [mean_of(BY[g], "certified", cert, "degree", "mean_degree") for g in GROUPS]
    deg[f"{label}: en büyük"] = [mean_of(BY[g], "certified", cert, "degree", "max_degree") for g in GROUPS]
    deg[f"{label}: derece<=5 %"] = [mean_of(BY[g], "certified", cert, "degree", "share_degree_le5", scale=100)
                                    for g in GROUPS]
    deg[f"{label}: çözümsüz %"] = [mean_of(BY[g], "certified", cert, "unresolved_share", scale=100) for g in GROUPS]
TABLES["3_kalan_graf"] = table(GROUPS, deg)
show("Tablo 3. Kanıtla elenemeyen kenarların grafı (düğüm derecesi; 'çözümsüz': kısmi dökümde karar verilemeyen düğümler)",
     TABLES["3_kalan_graf"].T, 2)

for cert, label in [("c2_lkh", "C2, ÜS = LKH turu"), ("c2_opt", "C2, ÜS = optimum"), ("c1_lkh", "C1, ÜS = LKH turu")]:
    cols = {lab: [mean_of(BY[g], "certified", cert, "lists5", name, "complete_share", scale=100) for g in GROUPS]
            for name, lab in LISTS}
    cols["olası en çok (derece<=5)"] = [mean_of(BY[g], "certified", cert, "degree", "share_degree_le5", scale=100)
                                        for g in GROUPS]
    TABLES[f"4_tam_liste_{cert}"] = table(GROUPS, cols)
    show(f"Tablo 4 ({label}). 5'li listesi kanıtla tam olan düğümler (%): düğümün kalan bütün kenarları listede",
         TABLES[f"4_tam_liste_{cert}"], 1)

cols = {}
for name, lab in LONG:
    cols[f"{lab}: medyan"] = [mean_of(BY[g], "certified", "c2_lkh", "long", name, "median") for g in GROUPS]
    cols[f"{lab}: %90"] = [mean_of(BY[g], "certified", "c2_lkh", "long", name, "p90") for g in GROUPS]
    cols[f"{lab}: <=10 %"] = [mean_of(BY[g], "certified", "c2_lkh", "long", name, "le10", scale=100) for g in GROUPS]
    cols[f"{lab}: listede yok %"] = [mean_of(BY[g], "certified", "c2_lkh", "long", name, "not_in_list", scale=100)
                                     for g in GROUPS]
TABLES["5_gereken_uzunluk"] = table(GROUPS, cols)
show("Tablo 5. Kanıtla tamlık için gereken liste uzunluğu (C2, ÜS = LKH turu; BetaTour ve sayma: 16 kenarlık graf)",
     TABLES["5_gereken_uzunluk"].T, 1)

TABLES["6_auc"] = table(GROUPS, {"sertifika": [",".join(sorted({r["auc"]["certificate"] for r in BY[g]})) for g in GROUPS],
                                 "BetaTour olasılığı": [mean_of(BY[g], "auc", "betatour") for g in GROUPS],
                                 "sayma sıklığı": [mean_of(BY[g], "auc", "count") for g in GROUPS],
                                 "-uzunluk": [mean_of(BY[g], "auc", "neg_distance") for g in GROUPS]})
show("Tablo 6. BetaTour grafındaki kenarlar: kalanları elenenlerden ayırma (AUC; 0,5 = rastgele)", TABLES["6_auc"])

TABLES["7_c2_denetimleri"] = table(GROUPS, {
    "denetim geçen": [f"{sum(bool(get(r, 'c2', 'checks', 'passed')) for r in BY[g])}/"
                      f"{sum(get(r, 'c2', 'checks') is not None for r in BY[g])}" for g in GROUPS],
    "en küçük ikili gevşeklik": [min([get(r, "c2", "checks", "dual_slack") for r in BY[g]
                                      if get(r, "c2", "checks") is not None] or [np.nan]) for g in GROUPS],
    "elenen opt kenarı (C2)": [sum(get(r, "c2", "checks", "opt_edges_eliminated") or 0 for r in BY[g]) for g in GROUPS],
    "elenen opt kenarı (C1)": [sum(get(r, "c1", "opt_edges_eliminated") or 0 for r in BY[g]) for g in GROUPS],
    "alfa-5 uyumu C1/LKH %": [mean_of(BY[g], "alpha_top5_agreement", scale=100) for g in GROUPS]})
show("Tablo 7. Sertifika denetimleri (kanıtlı optimuma karşı)", TABLES["7_c2_denetimleri"], 3)

# Karar
with_opt = [r for r in ROWS if r["opt"] is not None]
g1 = bool(C2_MODE is not None and with_opt and all(bool(get(r, "c2", "checks", "passed")) for r in with_opt))
have_c2 = all("c2_lkh" in r["certified"] for g in SYN for r in BY[g]) and all(BY[g] for g in SYN)
g2_detail = {g: mean_of(BY[g], "certified", "c2_lkh", "degree", "mean_degree") for g in SYN}
g2_unresolved = max(mean_of(BY[g], "certified", "c2_lkh", "unresolved_share") for g in SYN) if have_c2 else np.nan
g2 = bool(have_c2 and all(v <= G2_MAX_MEAN_DEGREE for v in g2_detail.values()) and g2_unresolved <= 0.01)
g3_detail = {}
for size in sorted({n for _, n, *_ in SYNTHETIC}):
    rs = [r for g, n, *_ in SYNTHETIC if n == size for r in BY[g]]
    share = {name: mean_of(rs, "certified", "c2_lkh", "lists5", name, "complete_share", scale=100)
             for name in ("betatour", "alpha_lkh", "count")}
    share["fark"] = share["betatour"] - max(share["alpha_lkh"], share["count"])
    g3_detail[str(size)] = share
g3 = bool(have_c2 and all(d["fark"] >= G3_MARGIN_PP for d in g3_detail.values()))
if not g1:
    verdict = ("C2 sertifikası denetimleri geçmedi ya da kullanılamadı: önce -z dökümünün anlamı çözülmeli. "
               "logs/ klasöründeki concorde_usage.txt ile selfcheck/selfcheck.json dosyalarını gönderin.")
elif not g2:
    verdict = ("Kök LP sertifikası grafı yeterince seyreltmiyor. Kanıtlı aday listeleri için daha güçlü kanıtlar "
               "(ör. Cook vd.'nin yerel eleme yöntemi) gerekir; fikir bu güçteki kanıtla yürümez.")
elif not g3:
    verdict = ("Kanıtlı seyrek graf elde edilebiliyor, ancak BetaTour basit sıralamalardan belirgin biçimde iyi bir "
               "önerici değil. Yöntem öğrenmeden kurulabilir ya da model bu hedefe göre yeniden eğitilmelidir.")
else:
    verdict = ("Devam: kanıtlı seyrek graf elde ediliyor ve BetaTour onu basit sıralamalardan iyi öneriyor. Sonraki "
               "adım: kanıtla tam listelerle LKH araması ve sertifikanın tam sayı aritmetiğiyle bağımsız denetimi.")
DECISION = {"G1_c2_checks": g1, "G2_sparse_graph": g2, "G2_mean_degree": g2_detail,
            "G2_max_unresolved_share": g2_unresolved, "G3_betatour_proposer": g3,
            "G3_shares_pct": g3_detail, "verdict": verdict, "c2_mode": C2_MODE}
print("\nKarar kuralı (pilottan önce yazıldı)")
print(f"  G1 C2 denetimleri: {'GEÇTİ' if g1 else 'GEÇMEDİ'}")
print(f"  G2 kalan grafın ortalama derecesi <= {G2_MAX_MEAN_DEGREE:g} (C2, ÜS = LKH): {'GEÇTİ' if g2 else 'GEÇMEDİ'} "
      + ", ".join(f"{g} {v:.2f}" for g, v in g2_detail.items())
      + f" (çözümsüz düğüm payı en çok %{100 * g2_unresolved:.2f}; sınır %1)")
print(f"  G3 BetaTour-5, alfa-5 ve sayma-5'i en az {G3_MARGIN_PP:g} puan aşar: {'GEÇTİ' if g3 else 'GEÇMEDİ'} "
      + "; ".join(f"{s} düğüm: BetaTour {d['betatour']:.1f}, alfa {d['alpha_lkh']:.1f}, sayma {d['count']:.1f}"
                  for s, d in g3_detail.items()))
print("  Sonuç:", verdict)

out_dir = OUTPUT_ROOT / "summary"
pd.json_normalize(ROWS).to_csv(out_dir / "instances.csv", index=False)
write_json(out_dir / "summary.json", {"decision": DECISION, "tables": {k: v.reset_index().to_dict(orient="records")
                                                                       for k, v in TABLES.items()},
                                      "selfcheck": SELF, "manifest": MANIFEST})
print(f"\nKaydedildi: {out_dir / 'summary.json'} ve {out_dir / 'instances.csv'}")
