"""Runs the listed arms of one probe instance concurrently, one LKH process per arm. 'uninterrupted' is a full
POPMUSIC-5 run; any other arm continues from the first-trial tour with the candidate file TAG/ARM.cand.

    python probe_arms.py LKH TAG LKH_SEED SECONDS ARM [ARM ...]
"""
import subprocess, sys, time
from pathlib import Path
LKH = sys.argv[1]
tag, seed, T, arms = sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), sys.argv[5:]
d = Path(tag)
procs = []
for arm in arms:
    par = d / f'{arm}.par'
    if arm == 'uninterrupted':
        par.write_text(f'PROBLEM_FILE = {d}/p.tsp\nCANDIDATE_SET_TYPE = POPMUSIC\nMAX_CANDIDATES = 5\nSUBGRADIENT = NO\nRUNS = 1\n'
                       f'MAX_TRIALS = 100000000\nSEED = {seed}\nTRACE_LEVEL = 2\nTIME_LIMIT = {T}\nSTOP_AT_OPTIMUM = NO\n')
    else:
        par.write_text(f'PROBLEM_FILE = {d}/p.tsp\nCANDIDATE_FILE = {d}/{arm}.cand\nPI_FILE = {d}/zero.pi\nMAX_CANDIDATES = 5\n'
                       f'SUBGRADIENT = NO\nINITIAL_TOUR_FILE = {d}/first.tour\nRUNS = 1\nMAX_TRIALS = 100000000\nSEED = {seed}\n'
                       f'TRACE_LEVEL = 2\nTIME_LIMIT = {T}\nSTOP_AT_OPTIMUM = NO\n')
    log = open(d / f'{arm}.log', 'w')
    procs.append(subprocess.Popen([LKH, str(par)], stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL))
for p in procs:
    p.wait()
print('done', arms)
