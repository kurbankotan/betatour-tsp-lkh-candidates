"""v11 Colab summary cell: mechanism measurements and guarded comparisons."""


def read_key_value_metrics(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        words = line.split()
        if len(words) != 2 or words[0] in values:
            raise ValueError(f"Bozuk telemetri satırı: {path}: {line!r}")
        values[words[0]] = float(words[1]) if "." in words[1] else int(words[1])
    return values


def comparison_summary(frame, treatment, control, basis):
    if frame.empty:
        return None
    rel = 100.0 * (frame[treatment].to_numpy(dtype=np.float64) -
                   frame[control].to_numpy(dtype=np.float64)) / frame[control].to_numpy(dtype=np.float64)
    return {"treatment": treatment, "control": control, "basis": basis,
            "n_instances": len(rel), "wins": int((rel < 0).sum()),
            "losses": int((rel > 0).sum()), "ties": int((rel == 0).sum()),
            "mean_relative_pct": float(rel.mean()),
            "median_relative_pct": float(np.median(rel))}


patched_arms = ["sham", "random_add", "frequency_add", "beta_add"]
guided_arms = ["random_add", "frequency_add", "beta_add"]
rows, mechanism_rows, target_rows, frequency_rows = [], [], [], []
summary_dir = OUTPUT_ROOT / "summary"
summary_dir.mkdir(exist_ok=True)

for inst in INSTANCES:
    records = {}
    for arm in ARMS:
        path = arm_record_path(inst, arm)
        if path.exists():
            records[arm] = validate_saved(inst, arm, json.loads(path.read_text()))
    if len(records) != len(ARMS):
        continue

    row = {"instance": inst["name"], "group": inst["group"],
           "n": inst["n"], "budget_s": inst["budget_s"],
           "mixed_hardware": len({json.dumps(records[arm]["hardware"], sort_keys=True)
                                   for arm in ARMS}) > 1}
    for arm in ARMS:
        row[arm] = records[arm]["best_cost"]
    row["first_tours_match"] = all(
        len({records[arm]["first_tour_hashes"][i] for arm in patched_arms}) == 1
        for i in range(len(SEEDS)))
    row["raw_first_tours_match"] = all(
        len({records[arm]["policy_audit"]["first_tour_raw_sha256"][i]
             for arm in patched_arms}) == 1 for i in range(len(SEEDS)))
    row["native_candidates_match"] = all(
        len({records[arm]["native_candidate_hashes"][i] for arm in patched_arms}) == 1
        for i in range(len(SEEDS)))
    row["live_candidates_match"] = all(
        len({records[arm]["live_candidate_hashes"][i] for arm in patched_arms}) == 1
        for i in range(len(SEEDS)))
    row["gates_match"] = all(
        len({records[arm]["policy_audit"]["gate_sha256_per_seed"][i]
             for arm in patched_arms}) == 1 for i in range(len(SEEDS)))
    row["model_predictions_match"] = len({
        records[arm]["policy_audit"]["model_probability_sha256"]
        for arm in patched_arms}) == 1
    row["planned_actions_match"] = len({
        json.dumps(records[arm]["policy_audit"]["all_guided_action_sha256_per_seed"],
                   sort_keys=True) for arm in patched_arms}) == 1
    row["capacities_match"] = all(
        len({records[arm]["policy_audit"]["selected_actions_per_seed"][i]
             for arm in guided_arms}) == 1 for i in range(len(SEEDS)))
    row["undirected_capacities_match"] = all(
        len({records[arm]["policy_audit"]["selected_unique_undirected_edges_per_seed"][i]
             for arm in guided_arms}) == 1 for i in range(len(SEEDS)))
    row["matched_policy_state"] = all(row[key] for key in
                                      ("first_tours_match", "raw_first_tours_match",
                                       "native_candidates_match", "live_candidates_match",
                                       "model_predictions_match", "planned_actions_match",
                                       "gates_match", "capacities_match",
                                       "undirected_capacities_match"))
    rows.append(row)

    for arm in patched_arms:
        record = records[arm]
        for index, seed in enumerate(SEEDS):
            base = OUTPUT_ROOT / record["audit_relpath"] / f"seed_{seed}"
            metrics = read_key_value_metrics(base / "telemetry.txt")
            selected = int(record["policy_audit"]["selected_actions_per_seed"][index])
            if int(metrics["action_count"]) != selected:
                raise AssertionError(f"Seçilen/uygulanan kenar sayısı farklı: {inst['name']}/{arm}/{seed}")
            mechanism_rows.append({"instance": inst["name"], "arm": arm,
                                   "seed": seed, "selected_actions": selected,
                                   "telemetry_age_at_end_s": max(0.0, min(
                                       float(record["deadline_epoch_s"]),
                                       float(record["start_epoch_s"]) + float(record["wall_elapsed_s"]))
                                       - float(metrics["telemetry_snapshot_epoch_s"])),
                                   **metrics})

    frequency_audit = records["beta_add"]["policy_audit"]
    for arm in guided_arms:
        for index, seed in enumerate(SEEDS):
            values = frequency_audit["all_guided_selected_ensemble_frequency_per_seed"][arm][index]
            frequency_rows.append({"instance": inst["name"], "arm": arm,
                                   "seed": seed, "selected_actions": len(values),
                                   "zero_frequency_actions": sum(value == 0 for value in values),
                                   "mean_ensemble_frequency": float(np.mean(values)) if values else np.nan})

    target = TARGET_COSTS[inst["name"]]
    for arm in ARMS:
        record = records[arm]
        event_map = (record["native_improvement_events"] if arm.startswith("native_")
                     else record["guided_improvement_events"])
        observed = [float(event["elapsed_s"])
                    for seed in SEEDS for event in event_map[str(seed)]
                    if int(event["cost"]) <= target]
        target_rows.append({"instance": inst["name"], "arm": arm,
                            "target_cost_from_v9_native_pop6": target,
                            "reached": bool(observed),
                            "earliest_elapsed_s": min(observed) if observed else np.nan,
                            "trace_kind": "approximate_native_file_poll" if arm.startswith("native_")
                            else "exact_patched_best_event"})

table = pd.DataFrame(rows)
mechanism_table = pd.DataFrame(mechanism_rows)
target_table = pd.DataFrame(target_rows)
frequency_table = pd.DataFrame(frequency_rows)
table.to_csv(summary_dir / "complete_instances.csv", index=False)
mechanism_table.to_csv(summary_dir / "candidate_mechanism.csv", index=False)
target_table.to_csv(summary_dir / "time_to_target_exploratory.csv", index=False)
frequency_table.to_csv(summary_dir / "selected_edge_frequencies.csv", index=False)

same_hardware = table.loc[~table["mixed_hardware"]].copy() if not table.empty else table
matched = same_hardware.loc[same_hardware["matched_policy_state"]].copy() if not same_hardware.empty else same_hardware
endpoint_rows = []
for control in ("native_pop6", "native_pop7", "sham", "random_add", "frequency_add"):
    result = comparison_summary(same_hardware, "beta_add", control,
                                "descriptive same-instance endpoint")
    if result is not None:
        endpoint_rows.append(result)
matched_rows = []
for control in ("sham", "random_add", "frequency_add"):
    result = comparison_summary(matched, "beta_add", control,
                                "all six raw first tours/candidate states/predictions/actions/gates/capacities match")
    if result is not None:
        matched_rows.append(result)
comparison_columns = ["treatment", "control", "basis", "n_instances", "wins",
                      "losses", "ties", "mean_relative_pct", "median_relative_pct"]
endpoint_table = pd.DataFrame(endpoint_rows, columns=comparison_columns)
matched_table = pd.DataFrame(matched_rows, columns=comparison_columns)
endpoint_table.to_csv(summary_dir / "endpoint_comparisons.csv", index=False)
matched_table.to_csv(summary_dir / "matched_policy_comparisons.csv", index=False)

if mechanism_table.empty:
    mechanism_aggregate = pd.DataFrame()
else:
    mechanism_aggregate = mechanism_table.groupby("arm", as_index=False)[
        ["selected_actions", "action_candidate_enumerations",
         "distinct_actions_enumerated", "action_accepted_edge_entries",
         "distinct_actions_entering_accepted_tours",
         "top_level_action_first_gate_passes", "top_level_action_positive_g1",
         "action_incumbent_entries", "distinct_actions_ever_in_better_tour"]].sum()
mechanism_aggregate.to_csv(summary_dir / "candidate_mechanism_totals.csv", index=False)

write_json_atomic(summary_dir / "pilot_note.json", {
    "analysis_revision": "v11-state-guarded-mechanism-1",
    "completed_instances": len(table), "planned_instances": len(INSTANCES),
    "same_hardware_instances": len(same_hardware),
    "matched_policy_instances": len(matched),
    "interpretation": "Reused-case mechanism pilot. Endpoint comparisons against native POP6/7 are descriptive. Patched policy comparisons require exact six-seed state, gate and capacity matching. Native time-to-target traces are approximate file polls; patched best.events are exact. No manuscript superiority claim."
})

print("Tamamlanan örnek:", len(table), "/", len(INSTANCES),
      "; tam eşleşmiş politika örneği:", len(matched))
display(table)
display(endpoint_table)
display(matched_table)
display(mechanism_aggregate)
display(frequency_table)
display(target_table)
