def save_progress(last_instance=None, state="running"):
    finished = sorted(p.stem for p in (OUTPUT_ROOT / "instances").glob("*.json"))
    write_json_atomic(OUTPUT_ROOT / "progress.json",
                      dict(state=state, completed=len(finished), total=len(INSTANCES), finished=finished,
                           last_instance=last_instance, updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))


save_progress(state="running")
session_start = time.time()
newly_finished = 0
for ordinal, inst in enumerate(INSTANCES):
    final_file = OUTPUT_ROOT / "instances" / (inst["name"] + ".json")
    if final_file.exists():
        record = json.loads(final_file.read_text())
        assert record.get("complete") and record.get("xy_sha256") == xy_hash(inst) \
            and set(record["results"]) == set(ARM_NAMES), f"{inst['name']}: eski kayıt bu kurulumla uyuşmuyor"
        print("ATLA (tamam):", inst["name"], flush=True)
        continue
    if MAX_NEW_INSTANCES is not None and newly_finished >= MAX_NEW_INSTANCES:
        print("Bu oturum için MAX_NEW_INSTANCES sınırına ulaşıldı.")
        break
    try:
        run_one_instance(inst, ordinal)
        newly_finished += 1
        save_progress(last_instance=inst["name"])
        done_now = len(list((OUTPUT_ROOT / "instances").glob("*.json")))
        left = sum(estimated_seconds(x) for x in INSTANCES
                   if not (OUTPUT_ROOT / "instances" / (x["name"] + ".json")).exists())
        print(f"  {done_now}/{len(INSTANCES)} örnek bitti; bu oturumda geçen süre "
              f"{(time.time() - session_start) / 60:.0f} dk, kalan yaklaşık {left / 3600:.1f} saat.", flush=True)
    except Exception as exc:
        failure = dict(instance=inst["name"], error=repr(exc), traceback=traceback.format_exc(),
                       utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        write_json_atomic(OUTPUT_ROOT / "errors" / (inst["name"] + ".json"), failure)
        save_progress(last_instance=inst["name"], state="error")
        raise

done = len(list((OUTPUT_ROOT / "instances").glob("*.json")))
save_progress(state="complete" if done == len(INSTANCES) else "partial")
print(f"Bu oturumda yeni {newly_finished}, toplam {done}/{len(INSTANCES)} örnek tamamlandı. "
      f"Süre: {(time.time() - session_start) / 3600:.2f} saat. Kayıt: {OUTPUT_ROOT}")
