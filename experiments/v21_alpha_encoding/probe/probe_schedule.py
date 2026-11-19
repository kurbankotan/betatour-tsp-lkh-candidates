"""Two-phase restart schedules. Phase A: the variant 'none' for TA seconds (its tour is saved). Phase B (TB seconds
each, run concurrently): each listed variant, restarted from the phase-A tour.

    python probe_schedule.py LKH TAG LKH_SEED TA TB VARIANT [VARIANT ...]
"""
import subprocess, sys
from pathlib import Path
LKH = sys.argv[1]
tag, seed, TA, TB, encs = sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), sys.argv[6:]
d = Path(tag)
def par(name, cand, init, T, out=None):
    p = d / f'{name}.par'
    p.write_text(f'PROBLEM_FILE = {d}/p.tsp\nCANDIDATE_FILE = {d}/{cand}.cand\nPI_FILE = {d}/zero.pi\nMAX_CANDIDATES = 5\nSUBGRADIENT = NO\n'
                 f'INITIAL_TOUR_FILE = {d}/{init}\nRUNS = 1\nMAX_TRIALS = 100000000\nSEED = {seed}\nTRACE_LEVEL = 2\nTIME_LIMIT = {T}\n'
                 f'STOP_AT_OPTIMUM = NO\n' + (f'OUTPUT_TOUR_FILE = {d}/{out}\n' if out else ''))
    return p
a = par('schedA', 'none', 'first.tour', TA, 'schedA.tour')
with open(d / 'schedA.log', 'w') as log:
    subprocess.run([LKH, str(a)], stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
procs = []
for e in encs:
    b = par(f'schedB_{e}', e, 'schedA.tour', TB)
    log = open(d / f'schedB_{e}.log', 'w')
    procs.append(subprocess.Popen([LKH, str(b)], stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL))
for p in procs: p.wait()
print('done', tag, encs)
