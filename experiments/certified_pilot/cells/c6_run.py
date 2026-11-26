# Deney (12 vCPU'da 35-70 dk): A) LKH turları ve listeleri, B) BetaTour, C) Concorde (kök LP ve kanıtlı optimum),
# D) örnek başına analiz. Oturum düşerse defteri baştan çalıştırın: biten adımlar Drive'dan okunur ve atlanır.
T0 = time.time()
INSTANCES = make_instances()
groups_seen = list(dict.fromkeys(i["group"] for i in INSTANCES))
say(f"{len(INSTANCES)} örnek: " + ", ".join(f"{g} {sum(i['group'] == g for i in INSTANCES)}" for g in groups_seen))

say("A) LKH: altı ilk deneme turu, alfa listeleri ve cezalar, üst sınır turu (tahmini 2-4 dk)")
run_parallel(stage_lkh, INSTANCES, CPU_COUNT, "A")
READY = [i for i in INSTANCES if (i["dir"] / "lkh.npz").exists()]

say(f"B) BetaTour çıkarımı, {DEVICE} (tahmini 1 dk)")
run_parallel(stage_model, READY, 1, "B")
READY = [i for i in READY if (i["dir"] / "model.npz").exists()]

say(f"C) Concorde: kök LP ve kanıtlı optimum, {max(1, CPU_COUNT - 1)} paralel süreç (tahmini 30-60 dk; uzun işler önce)")
jobs = []
for inst in READY:
    jobs.append(dict(name=inst["name"] + " (tam çözüm)", n=inst["n"], order=0, run=lambda inst=inst: job_full(inst)))
    jobs.append(dict(name=inst["name"] + " (kök LP)", n=inst["n"], order=1, run=lambda inst=inst: job_root(inst)))
jobs.sort(key=lambda j: (-j["n"], j["order"]))
run_parallel(lambda job: job["run"](), jobs, max(1, CPU_COUNT - 1), "C")

say("D) Analiz: C1 ve C2 sertifikaları, listeler, AUC (tahmini 2-4 dk)")
run_parallel(analyse, READY, 1, "D")
done = [i for i in READY if (i["dir"] / "analysis.json").exists()]
say(f"Bitti: {len(done)}/{len(INSTANCES)} örnek analiz edildi; toplam {(time.time() - T0) / 60:.1f} dk. "
    "Hatalar varsa: logs/errors.txt")
