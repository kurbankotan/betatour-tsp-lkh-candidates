# Ana adım. A) ilk denemeler, B) BetaTour ve listeler, C) bütün koşular tek rastgele sırada (PARALLEL süreç aynı
# anda). Biten her koşu Drive'a yazılır; defter yeniden çalıştırılınca kalanlardan devam eder.
T0 = time.time()
left_first = sum(not first_done(inst) for inst in INSTANCES)
say(f"A) İlk denemeler: {left_first} örnek x {len(SEEDS)} (tahmini {max(1, round(left_first * 6 * 2 / PARALLEL / 60))}"
    "-3 dk)")
stage_first(INSTANCES)
say(f"B) BetaTour ve listeler ({DEVICE}; tahmini 2-4 dk)")
for inst in INSTANCES:
    say(f"B: {inst['name']}: {stage_lists(inst)}")

JOBS = all_jobs()
todo = [job for job in JOBS if not run_path(BY_NAME[job[0]], job[1], job[2]).exists()]
cap_hours = sum(BY_NAME[name]["cap"] for name, _, _ in todo) / 3600
say(f"C) {len(todo)}/{len(JOBS)} koşu kaldı, aynı anda {PARALLEL} süreç. Üst sınırların toplamı {cap_hours:.1f} "
    f"çekirdek-saat; tahmini süre {0.35 * cap_hours / PARALLEL:.1f}-{0.7 * cap_hours / PARALLEL:.1f} saat "
    "(ilk 50 koşudan sonra yeniden tahmin edilir).")
t_c, finished, failed = time.time(), 0, 0
with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
    futures = {pool.submit(run_job, BY_NAME[name], arm, j): (name, arm, j) for name, arm, j in todo}
    for fut in as_completed(futures):
        name, arm, j = futures[fut]
        try:
            fut.result()
            finished += 1
        except Exception as exc:
            failed += 1
            log_error(f"{name}_{arm}_{SEEDS[j]}", exc)
            say(f"HATA: {name} {arm} {SEEDS[j]}: {exc!r}")
        done = finished + failed
        if done % 50 == 0 or done == len(todo):
            elapsed = time.time() - t_c
            eta = (len(todo) - done) * elapsed / max(done, 1)
            say(f"C: {done}/{len(todo)} koşu ({failed} hata); geçen {elapsed / 60:.0f} dk, kalan tahmini "
                f"{eta / 3600:.1f} saat")
missing = [job for job in JOBS if not run_path(BY_NAME[job[0]], job[1], job[2]).exists()]
say(f"Bitti: {len(JOBS) - len(missing)}/{len(JOBS)} koşu kayıtlı; bu oturum {(time.time() - T0) / 3600:.2f} saat."
    + (f" Eksik {len(missing)} koşu var: defteri yeniden çalıştırın (errors/ klasörüne bakın)." if missing else ""))
