"""Propose object groups for manual checking. Within each class, images stay in file-number order and are merged
bottom-up (average linkage) only with neighbours in that order, so every proposed object is a run of consecutive images.
Distance = 0.5 x cosine distance of deep features (object appearance) + 0.5 x Hellinger distance of the colour histogram
of the image border (background/surface). Usage: python 02_group.py <work_dir> <threshold> [class]
"""
import os, sys, glob, json, numpy as np
from PIL import Image
from scipy.sparse import diags
from sklearn.cluster import AgglomerativeClustering
work, T = sys.argv[1], float(sys.argv[2]); only = sys.argv[3] if len(sys.argv) > 3 else None
def border_hist(p):
    a = np.asarray(Image.open(p).convert('HSV').resize((60, 60)), np.float32)
    mask = np.ones((60, 60), bool); mask[12:48, 12:48] = False          # outer 20% ring = background
    h, _ = np.histogramdd(a[mask], bins=(12, 3, 3), range=((0, 256), (0, 256), (0, 256)))
    h = h.ravel(); return np.sqrt(h / h.sum())
res, tot = {}, 0
for p in sorted(glob.glob(os.path.join(work, 'feat', '*.npz')), key=str.lower):
    c = os.path.basename(p)[:-4]
    if only and c != only: continue
    z = np.load(p); files = list(z['files']); F = z['F']; F = F / np.linalg.norm(F, axis=1, keepdims=True)
    B = np.array([border_hist(os.path.join(work, 'thumbs', c, f)) for f in files])
    D = 0.5 * (1 - F @ F.T) + 0.5 * np.sqrt(np.clip(1 - B @ B.T, 0, None))
    n = len(files); conn = diags([1, 1], [-1, 1], shape=(n, n))
    lab = AgglomerativeClustering(n_clusters=None, distance_threshold=T, metric='precomputed', linkage='average', connectivity=conn).fit_predict(D)
    order, k, m = {}, 0, {}
    for i, l in enumerate(lab):
        if l not in order: k += 1; order[l] = k
        m[files[i]] = order[l]
    res[c] = m; sizes = np.bincount(list(m.values()))[1:]; tot += len(sizes)
    print(c.ljust(22), 'images', n, 'groups', len(sizes), 'median', int(np.median(sizes)), 'max', sizes.max(), 'singletons', int((sizes == 1).sum()))
print('TOTAL', tot)
json.dump(res, open(os.path.join(work, f'groups_T{T}.json'), 'w'))
