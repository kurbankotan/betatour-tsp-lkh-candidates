"""v11r2 Colab summary cell: mechanism measurements and guarded comparisons.

Primary endpoint (as in v9): the shortest valid tour among six seeds at the
deadline, compared by instance. Exact two-sided sign tests are descriptive
because the four instances were examined in earlier runs. Mean-of-six and
seed-level summaries, per-action mechanism rates and the trial-count ratio
against sham were added after the 2-vCPU v11 run and are descriptive.
"""

from math import comb


def read_key_value_metrics(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        words = line.split()
        if len(words) != 2 or words[0] in values:
            raise ValueError(f"Bozuk telemetri satırı: {path}: {line!r}")
        values[words[0]] = float(words[1]) if "." in words[1] else int(words[1])
    return values


def exact_sign_p(wins, losses):
    n, k = wins + losses, min(wins, losses)
    return 1.0 if n == 0 else min(1.0, 2.0 * sum(comb(n, j) for j in range(k + 1)) / 2 ** n)


def comparison_summary(frame, treatment, control, basis, column="{}"):
    if frame.empty:
        return None
    t = frame[column.format(treatment)].to_numpy(dtype=np.float64)
    c = frame[column.format(control)].to_numpy(dtype=np.float64)
    rel = 100.0 * (t - c) / c
    wins, losses = int((rel < 0).sum()), int((rel > 0).sum())
    return {"treatment": treatment, "control": control, "basis": basis,
            "n_instances": len(rel), "wins": wins, "losses": losses,
            "ties": int((rel == 0).sum()),
            "mean_relative_pct": float(rel.mean()),
            "median_relative_pct": float(np.median(rel)),
            "exact_sign_p_descriptive": exact_sign_p(wins, losses)}


patched_arms = ["sham", "random_add", "frequency_add", "beta_add"]
guided_arms = ["random_add", "frequency_add", "beta_add"]
rows, mechanism_rows, target_rows, frequency_rows, event_rows = [], [], [], [], []
seed_rows = []
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
        row["mean6_" + arm] = float(np.mean(records[arm]["seed_costs"]))
    audits = {arm: records[arm]["policy_audit"] for arm in patched_arms}
    row["first_tours_match"] = all(
        len({records[arm]["first_tour_hashes"][i] for arm in patched_arms}) == 1
        for i in range(len(SEEDS)))
    row["raw_first_tours_match"] = all(
        len({audits[arm]["first_tour_raw_sha256"][i] for arm in patched_arms}) == 1
        for i in range(len(SEEDS)))
    row["native_candidates_match"] = all(
        len({records[arm]["native_candidate_hashes"][i] for arm in patched_arms}) == 1
        for i in range(len(SEEDS)))
    row["live_candidates_match"] = all(
        len({records[arm]["live_candidate_hashes"][i] for arm in patched_arms}) == 1
        for i in range(len(SEEDS)))
    row["gates_match"] = all(
        len({audits[arm]["gate_sha256_per_seed"][i] for arm in patched_arms}) == 1
        for i in range(len(SEEDS)))
    row["model_predictions_match"] = len({
        audits[arm]["model_probability_sha256"] for arm in patched_arms}) == 1
    row["fresh_predictions_match"] = len({
        audits[arm]["fresh_model_probability_sha256"] for arm in patched_arms}) == 1
    row["fresh_inference_arms"] = sum(audits[arm]["decision_basis"] == "fresh_inference"
                                      for arm in patched_arms)
    differences = [audits[arm]["fresh_vs_decision_max_abs_probability_difference"]
                   for arm in patched_arms]
    row["max_fresh_vs_decision_probability_difference"] = (
        None if any(value is None for value in differences) else max(differences))
    row["planned_actions_match"] = len({
        json.dumps(audits[arm]["all_guided_action_sha256_per_seed"], sort_keys=True)
        for arm in patched_arms}) == 1
    row["capacities_match"] = all(
        len({audits[arm]["selected_actions_per_seed"][i] for arm in guided_arms}) == 1
        for i in range(len(SEEDS)))
    row["undirected_capacities_match"] = all(
        len({audits[arm]["selected_unique_undirected_edges_per_seed"][i]
             for arm in guided_arms}) == 1 for i in range(len(SEEDS)))
    row["matched_policy_state"] = all(row[key] for key in
                                      ("first_tours_match", "raw_first_tours_match",
                                       "native_candidates_match", "live_candidates_match",
                                       "model_predictions_match", "planned_actions_match",
                                       "gates_match", "capacities_match",
                                       "undirected_capacities_match"))
    rows.append(row)

    for index, seed in enumerate(SEEDS):
        seed_rows.append({"instance": inst["name"], "seed": seed,
                          **{arm: records[arm]["seed_costs"][index] for arm in ARMS}})

    for arm in patched_arms:
        record = records[arm]
        for index, seed in enumerate(SEEDS):
            base = OUTPUT_ROOT / record["audit_relpath"] / f"seed_{seed}"
            metrics = read_key_value_metrics(base / "telemetry.txt")
            applied = read_key_value_metrics(base / "applied.stats")
            selected = int(record["policy_audit"]["selected_actions_per_seed"][index])
            if int(metrics["action_count"]) != selected:
                raise AssertionError(f"Seçilen/uygulanan kenar sayısı farklı: {inst['name']}/{arm}/{seed}")
            events = record["guided_improvement_events"][str(seed)]
            mechanism_rows.append({"instance": inst["name"], "arm": arm,
                                   "seed": seed, "selected_actions": selected,
                                   "first_trial_end_s": events[0]["elapsed_s"],
                                   "barrier_wait_s": applied["wait_wall_s"],
                                   "telemetry_age_at_end_s": max(0.0, min(
                                       float(record["deadline_epoch_s"]),
                                       float(record["start_epoch_s"]) + float(record["wall_elapsed_s"]))
                                       - float(metrics["telemetry_snapshot_epoch_s"])),
                                   **metrics})

    for arm in ARMS:
        record = records[arm]
        event_map = (record["native_improvement_events"] if arm.startswith("native_")
                     else record["guided_improvement_events"])
        flags = record.get("event_flags") or {}
        for seed in SEEDS:
            trace = event_map[str(seed)]
            event_rows.append({"instance": inst["name"], "arm": arm, "seed": seed,
                               "events": len(trace),
                               "first_event_s": trace[0]["elapsed_s"],
                               "last_event_s": trace[-1]["elapsed_s"],
                               "events_after_deadline": sum(event["elapsed_s"] > inst["budget_s"]
                                                            for event in trace),
                               "final_tour_source": (record["final_tour_sources"][str(seed)]
                                                     if arm.startswith("native_") else "atomic_hook_best_tour"),
                               "flags": ";".join(flags.get(str(seed), []))})

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
                    if int(event["cost"]) <= target and float(event["elapsed_s"]) <= inst["budget_s"]]
        target_rows.append({"instance": inst["name"], "arm": arm,
                            "target_cost_v10_native_pop6_best_of_six": target,
                            "reached": bool(observed),
                            "earliest_elapsed_s": min(observed) if observed else np.nan,
                            "trace_kind": "approximate_native_file_poll" if arm.startswith("native_")
                            else "exact_patched_best_event"})

table = pd.DataFrame(rows)
seed_table = pd.DataFrame(seed_rows)
mechanism_table = pd.DataFrame(mechanism_rows)
target_table = pd.DataFrame(target_rows)
frequency_table = pd.DataFrame(frequency_rows)
event_table = pd.DataFrame(event_rows)
table.to_csv(summary_dir / "complete_instances.csv", index=False)
seed_table.to_csv(summary_dir / "seed_costs.csv", index=False)
mechanism_table.to_csv(summary_dir / "candidate_mechanism.csv", index=False)
target_table.to_csv(summary_dir / "time_to_target_exploratory.csv", index=False)
frequency_table.to_csv(summary_dir / "selected_edge_frequencies.csv", index=False)
event_table.to_csv(summary_dir / "improvement_trace_quality.csv", index=False)

same_hardware = table.loc[~table["mixed_hardware"]].copy() if not table.empty else table
matched = same_hardware.loc[same_hardware["matched_policy_state"]].copy() if not same_hardware.empty else same_hardware
comparison_columns = ["treatment", "control", "basis", "n_instances", "wins",
                      "losses", "ties", "mean_relative_pct", "median_relative_pct",
                      "exact_sign_p_descriptive"]
endpoint_rows, matched_rows, mean_rows = [], [], []
for control in ("native_pop6", "native_pop7", "sham", "random_add", "frequency_add"):
    result = comparison_summary(same_hardware, "beta_add", control,
                                "primary endpoint: best of six seeds")
    if result is not None:
        endpoint_rows.append(result)
    result = comparison_summary(same_hardware, "beta_add", control,
                                "descriptive: mean of six seed costs", "mean6_{}")
    if result is not None:
        mean_rows.append(result)
for treatment, control in (("sham", "native_pop6"), ("native_pop7", "native_pop6"),
                           ("frequency_add", "sham"), ("random_add", "sham"),
                           ("frequency_add", "random_add")):
    result = comparison_summary(same_hardware, treatment, control,
                                "context: best of six seeds")
    if result is not None:
        endpoint_rows.append(result)
for control in ("sham", "random_add", "frequency_add"):
    result = comparison_summary(matched, "beta_add", control,
                                "all six raw first tours/candidate states/decision inputs/actions/gates/capacities match")
    if result is not None:
        matched_rows.append(result)
endpoint_table = pd.DataFrame(endpoint_rows + mean_rows, columns=comparison_columns)
matched_table = pd.DataFrame(matched_rows, columns=comparison_columns)
endpoint_table.to_csv(summary_dir / "endpoint_comparisons.csv", index=False)
matched_table.to_csv(summary_dir / "matched_policy_comparisons.csv", index=False)

seed_level_rows = []
if not seed_table.empty:
    for treatment, control in (("beta_add", "sham"), ("beta_add", "random_add"),
                               ("beta_add", "frequency_add"), ("beta_add", "native_pop6")):
        rel = 100.0 * (seed_table[treatment] - seed_table[control]) / seed_table[control]
        seed_level_rows.append({"treatment": treatment, "control": control,
                                "seed_pairs": len(rel), "wins": int((rel < 0).sum()),
                                "losses": int((rel > 0).sum()), "ties": int((rel == 0).sum()),
                                "mean_relative_pct": float(rel.mean()),
                                "note": "descriptive; seeds are not independent units"})
seed_level_table = pd.DataFrame(seed_level_rows)
seed_level_table.to_csv(summary_dir / "seed_level_comparisons.csv", index=False)

mechanism_columns = ["selected_actions", "action_candidate_enumerations",
                     "distinct_actions_enumerated", "action_accepted_edge_entries",
                     "distinct_actions_entering_accepted_tours",
                     "top_level_action_enumerations",
                     "top_level_action_first_gate_passes", "top_level_action_positive_g1",
                     "action_incumbent_entries", "distinct_actions_ever_in_better_tour",
                     "current_better_tour_actions", "better_tour_events_after_action",
                     "trials_completed_after_action", "total_candidate_enumerations",
                     "slot_checks", "slot_restorations", "slot_restoration_displacement",
                     "slot_found_ahead", "slot_missing"]
if mechanism_table.empty:
    mechanism_aggregate = pd.DataFrame()
    rate_table = pd.DataFrame()
else:
    mechanism_aggregate = mechanism_table.groupby("arm", as_index=False)[mechanism_columns].sum()
    guided_totals = mechanism_aggregate.set_index("arm").loc[guided_arms]
    rate_table = pd.DataFrame({
        "enumerations_per_action": guided_totals.action_candidate_enumerations / guided_totals.selected_actions,
        "first_gate_passes_per_action": guided_totals.top_level_action_first_gate_passes / guided_totals.selected_actions,
        "accepted_entries_per_action": guided_totals.action_accepted_edge_entries / guided_totals.selected_actions,
        "pct_actions_ever_accepted": 100 * guided_totals.distinct_actions_entering_accepted_tours / guided_totals.selected_actions,
        "incumbent_entries_per_1000_actions": 1000 * guided_totals.action_incumbent_entries / guided_totals.selected_actions,
        "pct_actions_in_final_incumbent": 100 * guided_totals.current_better_tour_actions / guided_totals.selected_actions,
        "slot_restorations_per_action": guided_totals.slot_restorations / guided_totals.selected_actions,
    })
    sham_trials = mechanism_table[mechanism_table.arm == "sham"].set_index(["instance", "seed"])
    for arm in guided_arms:
        arm_trials = mechanism_table[mechanism_table.arm == arm].set_index(["instance", "seed"])
        ratio = (arm_trials.trials_completed_after_action /
                 sham_trials.trials_completed_after_action.replace(0, np.nan))
        rate_table.loc[arm, "median_trials_ratio_vs_sham"] = float(ratio.median())
    rate_table = rate_table.reset_index().rename(columns={"index": "arm"})
mechanism_aggregate.to_csv(summary_dir / "candidate_mechanism_totals.csv", index=False)
rate_table.to_csv(summary_dir / "candidate_mechanism_rates.csv", index=False)

write_json_atomic(summary_dir / "pilot_note.json", {
    "analysis_revision": "v11r2-state-guarded-mechanism-2",
    "completed_instances": len(table), "planned_instances": len(INSTANCES),
    "same_hardware_instances": len(same_hardware),
    "matched_policy_instances": len(matched),
    "hardware": HARDWARE,
    "interpretation": ("Reused-case mechanism pilot. Primary endpoint: best of six seeds by instance; "
                       "sign-test p-values are descriptive. Mean-of-six, seed-level and per-action "
                       "rates are descriptive. Patched policy comparisons require exact six-seed state, "
                       "decision input, gate and capacity matching. Native traces are approximate file "
                       "polls; patched best.events are exact. Incumbent entries are observational, not "
                       "causal. No manuscript superiority claim.")
})

print("Tamamlanan örnek:", len(table), "/", len(INSTANCES),
      "; tam eşleşmiş politika örneği:", len(matched))
display(table)
display(endpoint_table)
display(matched_table)
display(seed_level_table)
display(mechanism_aggregate)
display(rate_table)
display(frequency_table.groupby("arm")[["selected_actions", "zero_frequency_actions"]].sum()
        if not frequency_table.empty else frequency_table)
display(event_table.groupby("arm")[["events", "events_after_deadline"]].sum()
        if not event_table.empty else event_table)
display(target_table)
