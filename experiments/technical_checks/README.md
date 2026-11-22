# Technical checks (not runs)

Short checks on development data that decided whether a run was worth preparing. They are not runs of the study: they
have no protocol, use one development instance on a two-core machine, and their numbers are not reported in the
manuscript.

## After run v29: the time that staged merging leaves unused (22 November 2026)

**Question.** In run v29, staged merging stopped after 48 s of the 120 s left after the cut at 20,000 nodes on
average. Could that time, used by LKH-3 processes that continue from the staged tour and by a second merge, shorten the
tours further, and would BetaTour's candidate lists help there more than lists ranked by counting or the processes' own
POPMUSIC lists? This was the design planned as run v30.

**Data.** The uniform 20,000-node development instance of the run-v28 checks (drawn from the generator of the study;
not an instance of any run): its six recorded POPMUSIC-5 LKH-3 processes (seeds 201-206, 240 s) and the tour of staged
merging at half of the budget, 101,638,015, which is 0.068% shorter than the best tour at the cut (101,707,283).

**Lists.** `pop`: each process's own POPMUSIC-5 list. `cnt`: per node the three neighbours on BetaTour's graph that the
six first-trial tours use most often, then the process's POPMUSIC list, to five candidates, in the tree encoding of
run v21. `beta`: the same with BetaTour's ranking (the model read the six first-trial tours; on this machine's CPU the
inference took 26 s).

**Check 1.** For each list type, six LKH-3 processes (seeds 401-406) continued from the staged tour for 54 s each, and
one GA-EAX process then merged the staged tour and their best tours.

| Lists | Shortest tour of the six processes | After the merge | Change from the staged tour |
|---|---|---|---|
| `pop` | 101,637,799 | 101,637,799 | -0.00021% |
| `cnt` | 101,637,799 | 101,637,799 | -0.00021% |
| `beta` | 101,637,737 | 101,637,737 | -0.00027% |

Each process completed 36 to 100 trials. Twelve of the 18 processes found no shorter tour; five found the same
improvement of 216 units and one an improvement of 278 units (with the BetaTour lists).

**Check 2.** The processes continued for 30 s and kept their trial tours; the last 16 different trial tours of each
process went to a second staged merge (six GA-EAX groups with a 25 s limit, then the group results with the staged
tour and the processes' best tours, 12 s).

| Lists | Trial tours, mean distance from the staged tour | After the second merge | Change from the staged tour |
|---|---|---|---|
| `pop` | +0.135% | 101,637,707 | -0.00030% |
| `cnt` | +0.149% | 101,637,707 | -0.00030% |
| `beta` | +0.130% | 101,637,278 | -0.00073% |

**Reading.** The unused time shortened the staged tour by at most 0.0007%, about a hundredth of what staged merging
itself gained on this instance. The list types differed by a few hundred units on one instance; the difference of the
BetaTour lists in check 2 came from one of the six groups. Two processes ran at a time on the two cores, each with its
own time, so the six-process steps took three times as long as on the machine of the runs. Decision: run v30 was not
prepared, and BetaTour's lists in the phase after staged merging are not pursued further.
