"""v15 live-swap execution cell. Completed arms are immutable and resumable."""


def arm_record_path(inst, arm):
    return OUTPUT_ROOT / "instances" / inst["name"] / arm / "result.json"


def validate_saved(inst, arm, record):
    assert record["complete"] and record["arm"] == arm and record["instance"] == inst["name"]
    assert record["xy_sha256"] == inst["xy_sha256"] and record["seeds"] == SEEDS
    assert record["budget_s"] == inst["budget_s"] and record["build"] == BUILD_INFO
    assert record["code_sha256"] == CODE_SHA256
    assert record["fraction"] == SWAP_FRACTION and record["graph_k"] == GRAPH_K
    assert record["hardware_profile"] == HARDWARE_PROFILE
    assert len(record["seed_costs"]) == len(SEEDS)
    assert min(record["seed_costs"]) == record["best_cost"]
    if arm == "native_pop6":
        assert set(record["final_tour_sources"]) == {str(seed) for seed in SEEDS}
    else:
        assert len(record["first_tour_hashes"]) == len(SEEDS)
        assert len(record["live_candidate_hashes"]) == len(SEEDS)
        assert len(record["native_candidate_hashes"]) == len(SEEDS)
        assert len(record["protection_sha256"]) == 64
        audit = record["policy_audit"]
        assert len(audit["gate_sha256_per_seed"]) == len(SEEDS)
        assert audit["seed_ids"] == SEEDS
        assert audit["policy_version"] == POLICY_VERSION
        assert audit["decision_basis"] in ("fresh_inference", "cached_inference")
        if arm == "sham":
            assert audit["selected_actions_per_seed"] == [0] * len(SEEDS)
        else:
            assert audit["selected_actions_per_seed"] == audit["guided_actions_per_seed"]
        assert audit["selected_unique_undirected_edges_per_seed"] == audit["selected_actions_per_seed"]
    return record


def save_progress(state="running", last=None):
    completed = sum(arm_record_path(inst, arm).exists() for inst in INSTANCES for arm in ARMS)
    write_json_atomic(OUTPUT_ROOT / "progress.json",
                      {"state": state, "completed_arms": completed,
                       "total_arms": len(INSTANCES) * len(ARMS), "last": last,
                       "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})


new_arms = 0
save_progress()
for inst in INSTANCES:
    instance_dir = OUTPUT_ROOT / "instances" / inst["name"]
    instance_dir.mkdir(parents=True, exist_ok=True)
    coord_path = instance_dir / "xy.npz"
    if not coord_path.exists():
        temp = instance_dir / "xy.tmp.npz"
        np.savez_compressed(temp, xy=inst["xy"])
        os.replace(temp, coord_path)
    with np.load(coord_path, allow_pickle=False) as saved_xy:
        assert hashlib.sha256(np.ascontiguousarray(saved_xy["xy"]).tobytes()).hexdigest() == inst["xy_sha256"]
    local_problem = WORK_ROOT / "problems" / (inst["name"] + ".tsp")
    local_problem.parent.mkdir(exist_ok=True)
    T.write_tsp(local_problem, inst["xy"])
    for arm in arm_order(inst["name"]):
        record_path = arm_record_path(inst, arm)
        if record_path.exists():
            validate_saved(inst, arm, json.loads(record_path.read_text()))
            print("ATLA:", inst["name"], arm, flush=True)
            continue
        if MAX_NEW_ARMS is not None and new_arms >= MAX_NEW_ARMS:
            break
        attempt_id = uuid.uuid4().hex
        local_run = WORK_ROOT / "runs" / inst["name"] / arm / attempt_id
        local_run.parent.mkdir(parents=True, exist_ok=True)
        try:
            print("BAŞLA:", inst["name"], arm, "bütçe", inst["budget_s"], "sn", flush=True)
            if arm.startswith("native_"):
                raw = run_native_arm(arm, NATIVE_LKH, local_problem, inst["xy"],
                                     local_run, SEEDS, inst["budget_s"], T)
            else:
                # Arms whose model input is bitwise identical decide from the
                # first stored model output; every arm still runs inference.
                raw = run_guided_arm(arm, PATCHED_LKH, local_problem, inst["xy"],
                                     local_run, SEEDS, inst["budget_s"],
                                     MODEL, DEVICE, T, M,
                                     fraction=SWAP_FRACTION, graph_k=GRAPH_K,
                                     decision_cache_dir=instance_dir / "decision_cache",
                                     model_identity=CHECKPOINT_SHA256)
            audit_rel = f"instances/{inst['name']}/{arm}/{attempt_id}"
            audit_dest = OUTPUT_ROOT / audit_rel
            shutil.copytree(local_run, audit_dest)
            record = {"schema": 1, "complete": True, "instance": inst["name"],
                      "group": inst["group"], "n": inst["n"],
                      "xy_sha256": inst["xy_sha256"], "arm": arm,
                      "seeds": SEEDS, "budget_s": inst["budget_s"],
                      "seed_costs": raw["seed_costs"], "best_cost": raw["best_cost"],
                      "wall_elapsed_s": raw["wall_elapsed_s"],
                      "start_epoch_s": raw["start_epoch_s"],
                      "deadline_epoch_s": raw["deadline_epoch_s"] if "deadline_epoch_s" in raw else raw["start_epoch_s"] + inst["budget_s"],
                      "first_costs": raw.get("first_costs"),
                      "first_tour_hashes": raw.get("first_tour_hashes"),
                      "live_candidate_hashes": raw.get("live_candidate_hashes"),
                      "native_candidate_hashes": raw.get("native_candidate_hashes"),
                      "protection_sha256": raw.get("protection_sha256"),
                      "policy_audit": raw.get("policy_audit"),
                      "guided_improvement_events": raw.get("guided_improvement_events"),
                      "native_improvement_events": raw.get("native_improvement_events"),
                      "final_tour_sources": raw.get("final_tour_sources"),
                      "event_flags": raw.get("event_flags"),
                      "decision_cache_files_written": raw.get("decision_cache_files_written"),
                      "fraction": SWAP_FRACTION, "graph_k": GRAPH_K,
                      "build": BUILD_INFO, "code_sha256": CODE_SHA256,
                      "hardware": HARDWARE, "hardware_profile": HARDWARE_PROFILE,
                      "audit_relpath": audit_rel,
                      "attempt_id": attempt_id,
                      "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            validate_saved(inst, arm, record)
            write_json_atomic(record_path, record)
            new_arms += 1
            save_progress(last=inst["name"] + "/" + arm)
            print("BİTTİ:", inst["name"], arm, "en iyi", raw["best_cost"], flush=True)
        except Exception as exc:
            write_json_atomic(OUTPUT_ROOT / "errors" /
                              (inst["name"] + "_" + arm + "_" + attempt_id + ".json"),
                              {"instance": inst["name"], "arm": arm,
                               "error": repr(exc),
                               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            save_progress(state="error", last=inst["name"] + "/" + arm)
            raise
    if MAX_NEW_ARMS is not None and new_arms >= MAX_NEW_ARMS:
        break

completed_count = sum(arm_record_path(inst, arm).exists() for inst in INSTANCES for arm in ARMS)
save_progress(state="complete" if completed_count == len(INSTANCES) * len(ARMS) else "partial")
print(f"Tamamlanan kol: {completed_count}/{len(INSTANCES) * len(ARMS)}. "
      f"Bu oturumdaki yeni kol: {new_arms}")
