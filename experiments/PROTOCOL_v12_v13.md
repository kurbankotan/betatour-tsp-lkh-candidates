# Protocol for runs v12 and v13 (written before either run)

**Thesis under test.** Candidate coverage beyond POPMUSIC helps LKH-3 only where its search cannot recover the
missing edges by itself.

- **H1 (mechanism, run v12).** Within a run, LKH-3's candidate set repairs itself: AdjustCandidateSet appends the edges
  of every new best tour, so reference-tour edges missing from the initial POPMUSIC lists enter the live candidate set
  during the search.
- **H2 (positive branch, run v13).** On instances that are hard for LKH-3, BetaTour lists miss fewer optimal-tour edges
  than POPMUSIC and alpha-nearness lists.

Whatever their direction, the results of both runs are reported in the paper.

## Run v12: candidate self-correction (notebook 13)

- **Instances.** The 43 instances of the earlier LKH experiments that have a reference tour in the coverage-study
  data (16 x 1,000 nodes, 16 x 10,000 nodes, 11 TSPLIB), with the budgets of Table 1 of the paper. Instances whose
  reference tour is missing or whose coordinates differ from the regenerated instance are excluded and listed.
- **Runs.** Six native, ascent-free POPMUSIC-5 LKH-3.0.14 processes (seeds 101-106) in parallel per instance, as the
  uninterrupted baseline of runs v4-v8: MAX_CANDIDATES = 5, SUBGRADIENT = NO, RUNS = 1, MAX_TRIALS = 10^8,
  TRACE_LEVEL = 1, one external monotonic deadline at the budget. No source patch. Each process writes its initial
  candidate lists C0 (CANDIDATE_FILE) and rewrites OUTPUT_TOUR_FILE after every new best tour; a 20 ms poller copies
  every completed rewrite. Six separate single-trial runs with the same seeds give the first-trial tours, the M = 6
  ensemble of the BetaTour and counting top-5 lists (runs v4 and v5).
- **Measures (per 1,000 nodes, seed means per instance).** Reference edges missing from C0, from the BetaTour top-5
  list, from the counting top-5 list, and from the dynamic set C0 + all edges of every best tour found by 25, 50, 75 and
  100% of the budget; the share of the edges missing from C0 that the search reached; the share of these edges and of
  the edges present in C0 that the final tour contains; reference edges in the BetaTour lists that the search never
  reached (an upper bound of what the lists could add). The poller can miss a best tour overwritten within one
  interval, so the dynamic set is a lower bound of LKH's live candidate set.
- **Primary comparison.** Per instance, reference edges missing from the dynamic set at the budget versus from the
  BetaTour top-5 list; exact two-sided sign test over instances. H1 is supported if the dynamic set misses fewer edges
  on more instances and p < 0.05. Group means are descriptive.
- **Descriptive check.** Best-of-six final lengths are compared with the uninterrupted POPMUSIC arm of run v5. The
  deadlines differ (run v5 used LKH's TIME_LIMIT and an external stop 3 s later), so equality is not expected.

## Run v13: coverage on hard instances (notebook 14, written after the instance data have been inspected)

- **Sets.** S1: the clustered, morphed and random uniform instances of the TSP algorithm-selection study with their
  Concorde optimal lengths (primary); S2: the hard instances published with "The Suffering Salesperson" (PPSN 2026);
  S3: TSPLIB instances of 1,000-2,392 nodes with their known optimal lengths. Optimal tours: LKH-3 runs until the known
  optimal length is reached (cap 300 s); instances without a verified optimal tour are excluded and counted.
- **Hardness screen.** Ascent-free POPMUSIC-5 LKH-3 with a 10 s limit; an instance that does not reach its optimal
  length within 10 s is LKH-hard. Sample: all LKH-hard S1 instances (at most 200, drawn at random if more) and 100
  random others.
- **Lists (first five per node).** alpha-nearness after ascent; POPMUSIC-5; counting-8 and BetaTour-8 (the same
  eight single-trial tours, extended graph); POP-8 union (eight POPMUSIC sets from different seeds, ranked by
  frequency, the information available to POPMUSIC with restarts).
- **Measure.** Optimal-tour edges missing per instance; for the full lists also missing at any position; share of
  instances with no missing edge; construction times.
- **Primary comparison.** On the LKH-hard S1 instances, BetaTour-8 versus POPMUSIC-5, paired by instance, exact two-sided
  sign test, p < 0.05 and a lower mean. Secondary: alpha, counting-8 and POP-8 union. S2 and S3 are descriptive.
- **Gate for a solver experiment (run v14).** The primary comparison is positive and BetaTour-8 also misses fewer
  optimal edges than the POP-8 union (sign test, p < 0.05). Otherwise the positive branch closes and the result is
  reported.

## Common rules

LKH-3.0.14 is pinned and hashed; the BetaTour checkpoint is pinned by its SHA-256
(2322f66c752913f6179f8c61f6b50b0f9d21426857d3ec0155e02891fb3ebd70); the experiment code is embedded in each
notebook, copied to its result archive and hashed in its manifest; a changed setting or code requires a new
experiment name.
