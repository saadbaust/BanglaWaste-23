"""Apply the team's review edits to the proposed groups.
Usage: python 05_apply_edits.py <proposed_groups_json> <review_edits_json> <out_final_json>"""
import sys, json, re
P = json.load(open(sys.argv[1])); E = json.load(open(sys.argv[2]))
num = lambda f: int(re.search(r'_(\d{5})\.jpg$', f).group(1))
final = {c: dict(m) for c, m in P.items()}
for c, objs in E.items():
    if c.startswith('_'): continue
    byn = {num(f): f for f in final[c]}
    used = set()
    for oid, nums in objs.items():
        g = int(oid[3:])
        if g in set(P[c].values()): sys.exit(f'{c} {oid}: ID already used by a proposed object')
        for n in nums:
            if n not in byn: sys.exit(f'{c}: no image number {n}')
            if n in used: sys.exit(f'{c}: image {n} listed twice')
            used.add(n); final[c][byn[n]] = g
json.dump(final, open(sys.argv[3], 'w'))
for c in E:
    if c.startswith('_'): continue
    print(c, 'objects before', len(set(P[c].values())), 'after', len(set(final[c].values())), 'images moved', sum(len(v) for v in E[c].values()))
