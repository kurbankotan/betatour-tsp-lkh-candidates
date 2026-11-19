# Exploratory probe that preceded the protocol of run v21

After `ChooseInitialTour.c` had been read (the restart of every trial after the first keeps a best-tour edge only if
it is a candidate with alpha = 0), this probe checked whether the alpha-values of a candidate file change a restarted
LKH-3 search. It ran in the development sandbox (two virtual CPUs, LKH-3.0.14 built from the pinned archive) on
19 November 2026. It is exploratory: the number of instances grew in three steps (1, 8 and 9) and the comparisons were
read between the steps. Its results were used to design run v21 and are not a confirmation of anything.

## Design

- 18 uniform instances of 10,000 nodes with integer coordinates below 10^6 (`random.Random`; instance seed 7 for s101
  and 7s + 1 for s102 to s118; LKH seed s).
- First phase: ascent-free POPMUSIC-5 LKH-3 with one trial; LKH writes its candidate file and its tour.
- Variants of that file. All keep the lists and the fathers of LKH's file and change the alpha-values only:
  `native` (LKH's own), `rank` (0, 100, 200, ...), `top2` (0 for the first two entries), `zero` (0 for every entry),
  `none` (no entry 0).
- Arms: the uninterrupted run with the same seed and one continuation per variant from the first-trial tour (zero
  penalties), 60 s each, one LKH process per arm. s101 to s109 ran all six arms as three concurrent pairs, s110 to
  s118 ran `uninterrupted`, `native` and `rank` as three concurrent processes.
- Clock: LKH's own CPU time since the search of each process began; the uninterrupted run is shifted by the end of
  its first trial. Reading the files and starting the process are not charged, so this is not the time accounting of
  the paper's runs.
- Two-phase schedule (s101 to s109): 30 s with `none`, then 29 s with `zero` or `none` from the resulting tour.

## Results

Mean percentage difference of the best tour length; in brackets the number of instances on which the first arm was
longer / equal / shorter and, where below 0.05, the p-value of the exact sign test.

| First arm minus second arm (18 instances) | after 10 s | after 30 s | after 59 s |
|---|---|---|---|
| `rank` - `native` | +0.021 (14/0/4, p = 0.031) | +0.011 (11/0/7) | +0.007 (12/0/6) |
| `rank` - `uninterrupted` | +0.016 (11/0/7) | +0.012 (14/0/4, p = 0.031) | +0.006 (11/0/7) |
| `native` - `uninterrupted` | -0.005 (10/0/8) | +0.001 (11/0/7) | -0.002 (9/0/9) |

The 95% bootstrap interval of `rank` - `native` after 59 s is -0.001 to +0.015. Trials completed in 60 s: 248
(`uninterrupted`), 271 (`native`), 171 (`rank`). Share of the first-trial tour's edges with alpha = 0: 0.757
(`native`), 0.444 (`rank`); 17 instances, the set-up output of s101 was not kept.

Nine instances with every variant, difference to `native` after 59 s: `top2` +0.004 (5/0/4), `none` +0.004 (5/0/4),
`zero` +0.034 (9/0/0, p = 0.004). Two-phase schedule, `zero` minus `none` in the second phase: +0.006 (7/0/2).

With the rank encoding the restarted search completed fewer trials and was behind early; the difference shrank with
time and its interval included zero after 59 s. Marking every candidate with alpha = 0 was worse than LKH's own
marking on all nine instances.

## Limits

One family and size, one process per arm, CPU clock, concurrent processes on two CPUs (three for s110 to s118),
growing sample. The files of runs v1 to v8 also lacked the fathers; the probe did not measure that field.

## Files

`probe_setup.py`, `probe_arms.py`, `probe_schedule.py` and `run_probe.sh` are the scripts of the probe with the LKH
path as an argument. `probe_summary.py` reads the LKH logs and wrote `probe_results.csv` (one row per instance and
arm: trials and best length after 5 to 59 s) and `probe_summary.txt`. The logs (61 MB) are not in the repository.
