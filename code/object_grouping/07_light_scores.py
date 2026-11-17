"""Score every image for signs of phone-flash lighting (visual cues only, from the review thumbnails).
clip    = share of near-white, low-saturation pixels (specular hot spots)
falloff = mean brightness of the centre / mean brightness of the outer ring (flash lights the centre)
Usage: python 07_light_scores.py <work_dir> <groups_json> <out_csv>"""
import sys, os, json, csv
import numpy as np
from PIL import Image
work, gj, out = sys.argv[1:4]; G = json.load(open(gj))
rows = []
for c in sorted(G, key=str.lower):
    for f, g in G[c].items():
        a = np.asarray(Image.open(os.path.join(work, 'thumbs', c, f)).convert('HSV').resize((96, 96)), np.float32)
        S, V = a[..., 1], a[..., 2]
        clip = float(((V >= 245) & (S <= 40)).mean())
        ring = np.ones((96, 96), bool); ring[16:80, 16:80] = False
        falloff = float(V[30:66, 30:66].mean() / max(V[ring].mean(), 1))
        rows.append([c, f, g, round(clip, 4), round(falloff, 3), round(float(V.mean()), 1)])
with open(out, 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['class_folder', 'file_name', 'object', 'clip', 'falloff', 'mean_v']); w.writerows(rows)
print(len(rows))
