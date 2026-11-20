import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["group"] == "tsplib"]
print(f"{'instance':10s} {'n':>6s} {'budget':>7s} | best6 gap%: pop_full  beta_tree | mean6 gap%: pop_full  beta_tree")
acc = {k: [] for k in ("pb", "bb", "pm", "bm")}
big = {k: [] for k in ("pb", "bb", "pm", "bm")}
for r in rows:
    opt = float(r["optimum"]); n = int(r["n"])
    g = lambda v: 100.0 * (float(v) - opt) / opt
    v = dict(pb=g(r["pop_full"]), bb=g(r["beta_tree"]), pm=g(r["pop_full_mean6"]), bm=g(r["beta_tree_mean6"]))
    for k in acc:
        acc[k].append(v[k])
        if n >= 7000:
            big[k].append(v[k])
    print(f"{r['instance']:10s} {n:6d} {float(r['budget_s']):7.1f} | {v['pb']:19.4f} {v['bb']:10.4f} | {v['pm']:19.4f} {v['bm']:10.4f}")
m = lambda a: sum(a) / len(a)
print("all  ", len(acc["pb"]), {k: round(m(a), 4) for k, a in acc.items()})
print("n>=7k", len(big["pb"]), {k: round(m(a), 4) for k, a in big.items()})
