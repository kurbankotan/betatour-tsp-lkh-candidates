"""v15 live-swap exploratory analysis on paired development instances.

Only complete same-hardware instances contribute to endpoint comparisons.
Policy comparisons additionally require exact first tours, candidate states,
model decision arrays, source gates, victims and action capacities.  The four
instances were examined in previous runs, so p-values are descriptive only.
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


def compare(frame, treatment, control, basis, column="{}"):
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


patched_arms = ["sham", "random_swap", "nearest_swap", "beta_swap"]
guided_arms = patched_arms[1:]
rows, seed_rows, mechanism_rows = [], [], []
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
    audits = {arm: records[arm]["policy_audit"] for arm in patched_arms}
    row = {"instance": inst["name"], "group": inst["group"],
           "n": inst["n"], "budget_s": inst["budget_s"],
           "mixed_hardware": len({json.dumps(records[arm]["hardware_profile"], sort_keys=True)
                                  for arm in ARMS}) != 1}
    for arm in ARMS:
        row[arm] = records[arm]["best_cost"]
        row["mean6_" + arm] = float(np.mean(records[arm]["seed_costs"]))
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
    row["protected_edges_match"] = len({records[arm]["protection_sha256"]
                                         for arm in patched_arms}) == 1
    row["gates_match"] = all(
        len({audits[arm]["gate_sha256_per_seed"][i] for arm in patched_arms}) == 1
        for i in range(len(SEEDS)))
    row["victims_match"] = all(
        len({audits[arm]["victim_sha256_per_seed"][i] for arm in patched_arms}) == 1
        for i in range(len(SEEDS)))
    row["model_predictions_match"] = len({audits[arm]["model_probability_sha256"]
                                           for arm in patched_arms}) == 1
    row["planned_actions_match"] = len({
        json.dumps(audits[arm]["all_guided_action_sha256_per_seed"], sort_keys=True)
        for arm in patched_arms}) == 1
    row["capacities_match"] = all(
        len({audits[arm]["selected_actions_per_seed"][i] for arm in guided_arms}) == 1
        for i in range(len(SEEDS)))
    row["matched_policy_state"] = all(row[key] for key in (
        "first_tours_match", "raw_first_tours_match", "native_candidates_match",
        "live_candidates_match", "protected_edges_match", "gates_match",
        "victims_match", "model_predictions_match", "planned_actions_match",
        "capacities_match"))
    rows.append(row)

    for i, seed in enumerate(SEEDS):
        seed_rows.append({"instance": inst["name"], "seed": seed,
                          **{arm: records[arm]["seed_costs"][i] for arm in ARMS}})
    for arm in patched_arms:
        for i, seed in enumerate(SEEDS):
            base = OUTPUT_ROOT / records[arm]["audit_relpath"] / f"seed_{seed}"
            metrics = read_key_value_metrics(base / "telemetry.txt")
            applied = read_key_value_metrics(base / "applied.stats")
            selected = int(audits[arm]["selected_actions_per_seed"][i])
            assert int(metrics["action_count"]) == selected
            assert int(applied["swapped"]) == selected
            assert int(applied["rejected"]) == 0
            assert int(applied["row_length_preserved"]) == 1
            mechanism_rows.append({"instance": inst["name"], "arm": arm,
                                   "seed": seed, "selected_actions": selected,
                                   "barrier_wait_s": float(applied["wait_wall_s"]),
                                   "telemetry_age_at_end_s": max(0.0, min(
                                       float(records[arm]["deadline_epoch_s"]),
                                       float(records[arm]["start_epoch_s"]) +
                                       float(records[arm]["wall_elapsed_s"])) -
                                       float(metrics["telemetry_snapshot_epoch_s"])),
                                   **metrics})

table = pd.DataFrame(rows)
seed_table = pd.DataFrame(seed_rows)
mechanism_table = pd.DataFrame(mechanism_rows)
table.to_csv(summary_dir / "complete_instances.csv", index=False)
seed_table.to_csv(summary_dir / "seed_costs.csv", index=False)
mechanism_table.to_csv(summary_dir / "swap_mechanism.csv", index=False)

same_hardware = table.loc[~table["mixed_hardware"]].copy() if not table.empty else table
matched = same_hardware.loc[same_hardware["matched_policy_state"]].copy() if not same_hardware.empty else same_hardware
comparisons = []
for control in ("native_pop6", "sham", "nearest_swap", "random_swap"):
    basis = "same hardware, best-of-six endpoint"
    frame = same_hardware if control == "native_pop6" else matched
    result = compare(frame, "beta_swap", control, basis)
    if result:
        comparisons.append(result)
    result = compare(frame, "beta_swap", control, "descriptive mean-of-six", "mean6_{}")
    if result:
        comparisons.append(result)
comparison_table = pd.DataFrame(comparisons)
comparison_table.to_csv(summary_dir / "exploratory_comparisons.csv", index=False)

mechanism_summary = pd.DataFrame()
if not mechanism_table.empty:
    measures = [name for name in (
        "selected_actions", "action_candidate_enumerations",
        "action_accepted_edge_entries", "action_incumbent_entries",
        "trials_completed_after_action", "total_candidate_enumerations")
        if name in mechanism_table.columns]
    mechanism_summary = mechanism_table.groupby("arm", as_index=False)[measures].sum()
mechanism_summary.to_csv(summary_dir / "swap_mechanism_totals.csv", index=False)

write_json_atomic(summary_dir / "pilot_note.json", {
    "completed_instances": len(table), "planned_instances": len(INSTANCES),
    "same_hardware_instances": len(same_hardware),
    "matched_policy_instances": len(matched),
    "interpretation": "Reused-case exploratory pilot. Instance is the analysis unit; seeds are not independent. Sign p-values are descriptive. Native POP6 is the uninterrupted practical benchmark. Directed row length is fixed only at the swap barrier; LKH may append or reorder candidates later. Reverse-row presence of evicted edges can leave the undirected graph size unchanged or increase it. No superiority claim before disjoint holdout."})

print("Tamamlanan örnek:", len(table), "/", len(INSTANCES),
      "; eşlenmiş politika örneği:", len(matched))
display(table)
display(comparison_table)
display(mechanism_summary)
