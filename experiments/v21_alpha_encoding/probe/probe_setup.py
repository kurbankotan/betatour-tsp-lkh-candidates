"""Phase 1 of the alpha-encoding probe: instance, LKH's own POPMUSIC-5 candidate file and first-trial tour, then the
variants of the file. Every variant keeps the fathers of LKH's file and changes the alpha-values only.

    python probe_setup.py LKH NODES INSTANCE_SEED LKH_SEED TAG      (writes the directory TAG)
"""
import subprocess, sys, time, random
from pathlib import Path
LKH = sys.argv[1]
n, inst_seed, lkh_seed, tag = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
d = Path(tag); d.mkdir(exist_ok=True)
rng = random.Random(inst_seed)
pts = [(rng.randrange(1_000_000), rng.randrange(1_000_000)) for _ in range(n)]
(d / 'p.tsp').write_text('NAME : probe\nTYPE : TSP\nDIMENSION : %d\nEDGE_WEIGHT_TYPE : EUC_2D\nNODE_COORD_SECTION\n' % n
                         + ''.join(f'{i} {x} {y}\n' for i, (x, y) in enumerate(pts, 1)) + 'EOF\n')
(d / 'zero.pi').write_text(f'{n}\n' + ''.join(f'{i} 0\n' for i in range(1, n + 1)) + '-1\nEOF\n')
(d / 'phase1.par').write_text(f'PROBLEM_FILE = {d}/p.tsp\nCANDIDATE_SET_TYPE = POPMUSIC\nMAX_CANDIDATES = 5\nSUBGRADIENT = NO\n'
                              f'RUNS = 1\nMAX_TRIALS = 1\nSEED = {lkh_seed}\nTRACE_LEVEL = 1\nCANDIDATE_FILE = {d}/native.cand\n'
                              f'OUTPUT_TOUR_FILE = {d}/first.tour\n')
t = time.time()
out = subprocess.run([LKH, str(d / 'phase1.par')], capture_output=True, text=True, stdin=subprocess.DEVNULL).stdout
(d / 'phase1.log').write_text(out)
print('phase 1 wall', round(time.time() - t, 1), 's')
print('\n'.join(l for l in out.splitlines() if 'Preprocessing' in l or 'Cost' in l or 'Time' in l)[-600:])
# parse native candidate file and write the variants
lines = (d / 'native.cand').read_text().split('\n')
assert int(lines[0]) == n
rows = []
for line in lines[1:n + 1]:
    f = list(map(int, line.split()))
    node, dad, cnt = f[0], f[1], f[2]
    pairs = [(f[3 + 2 * k], f[4 + 2 * k]) for k in range(cnt)]
    rows.append((node, dad, pairs))
def write(name, enc):
    with open(d / name, 'w') as fh:
        fh.write(f'{n}\n')
        for node, dad, pairs in rows:
            fh.write(f'{node} {dad} {len(pairs)} ' + ' '.join(f'{to} {enc(k, a)}' for k, (to, a) in enumerate(pairs)) + '\n')
        fh.write('-1\nEOF\n')
write('rank.cand', lambda k, a: 100 * k)            # as in the paper's restarted arms
write('top2.cand', lambda k, a: 0 if k < 2 else 100 * (k - 1))
write('zero.cand', lambda k, a: 0)
write('none.cand', lambda k, a: 100 * (k + 1))      # no candidate trusted
# statistics of alpha == 0
tour = [int(x) for x in (d / 'first.tour').read_text().split('TOUR_SECTION')[1].split() if x.lstrip('-').isdigit()]
tour = [x for x in tour if x > 0]
assert len(tour) == n
suc = {a: b for a, b in zip(tour, tour[1:] + tour[:1])}
a0 = {node: {to for to, a in pairs if a == 0} for node, _, pairs in rows}
first = {node: pairs[0][0] for node, _, pairs in rows}
top2 = {node: {to for to, _ in pairs[:2]} for node, _, pairs in rows}
cand = {node: {to for to, _ in pairs} for node, _, pairs in rows}
print('candidates per node %.2f; alpha==0 per node (native) %.2f' % (sum(len(p) for _, _, p in rows) / n, sum(len(v) for v in a0.values()) / n))
for lab, trusted in (('native alpha==0', a0), ('rank-1 only', {k: {v} for k, v in first.items()}), ('top-2', top2), ('all candidates', cand)):
    fwd = sum(suc[a] in trusted[a] for a in tour) / n
    bwd = sum(a in trusted[suc[a]] for a in tour) / n
    print(f'  first-tour edges trusted ({lab}): forward {fwd:.3f}, backward {bwd:.3f}')
