"""Merge the fine groups into 5-10 objects per class.
Fine groups (incl. team edits) are kept whole and ordered by their first image number. Only neighbouring
groups in that order are merged (photos of one item were taken in a run), using Ward's criterion on
MobileNetV2 features, which merges the most similar neighbours first and avoids one giant group.
The number of objects per class (5-10) is set where the merge cost jumps most.
Usage: python 06_coarse.py <work_dir> <fine_groups_json> <out_json> [kmin] [kmax]"""
import sys, os, json, re
import numpy as np
work, fj, out = sys.argv[1:4]; kmin = int(sys.argv[4]) if len(sys.argv) > 4 else 5; kmax = int(sys.argv[5]) if len(sys.argv) > 5 else 10
FG = json.load(open(fj)); num = lambda f: int(re.search(r'_(\d{5})\.jpg$', f).group(1))
res = {}; tot = 0
for c in sorted(FG, key=str.lower):
    z = np.load(os.path.join(work, 'feat', f'{c}.npz'), allow_pickle=True)
    files = list(z['files']); F = z['F'].astype(np.float64); F /= np.linalg.norm(F, axis=1, keepdims=True)
    idx = {f: i for i, f in enumerate(files)}
    first = {}
    for f in sorted(FG[c], key=num): first.setdefault(FG[c][f], num(f))
    segs = sorted(first, key=first.get)                       # fine groups in shooting order
    cl = [[s] for s in segs]
    S = {s: F[[idx[f] for f in FG[c] if FG[c][f] == s]] for s in segs}
    stat = lambda grp: (sum(len(S[s]) for s in grp), np.vstack([S[s] for s in grp]).mean(0))
    st = [stat(g) for g in cl]; cost = {}
    while len(cl) > 1:
        w = [(na * nb / (na + nb)) * np.sum((ma - mb) ** 2) for (na, ma), (nb, mb) in zip(st[:-1], st[1:])]
        i = int(np.argmin(w)); cost[len(cl)] = w[i]
        if len(cl) == kmin - 1: break
        cl[i] = cl[i] + cl.pop(i + 1); st[i] = stat(cl[i]); st.pop(i + 1)
        if len(cl) == kmin - 1: break
    # redo cleanly to the chosen k
    n = len(segs)
    k = n if n <= kmin else max(range(kmin, min(kmax, n) + 1), key=lambda k: cost.get(k, 0) / max(cost.get(k + 1, 1e-9), 1e-9))
    cl = [[s] for s in segs]; st = [stat(g) for g in cl]
    while len(cl) > k:
        w = [(na * nb / (na + nb)) * np.sum((ma - mb) ** 2) for (na, ma), (nb, mb) in zip(st[:-1], st[1:])]
        i = int(np.argmin(w)); cl[i] = cl[i] + cl.pop(i + 1); st[i] = stat(cl[i]); st.pop(i + 1)
    lab = {s: j + 1 for j, g in enumerate(cl) for s in g}
    m = {f: lab[FG[c][f]] for f in FG[c]}
    res[c] = m; tot += k
    sizes = [list(m.values()).count(g) for g in range(1, k + 1)]
    print(f'{c:22s} fine {n:3d} -> {k:2d} objects  sizes {sizes}')
print('TOTAL', tot)
json.dump(res, open(out, 'w'))
