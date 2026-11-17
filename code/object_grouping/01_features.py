"""Extract MobileNetV2 (ImageNet, ONNX) features and small thumbnails for every image, class by class.
Used only to PROPOSE object groups for manual checking. Usage: python 01_features.py <dataset_root> <work_dir> <time_budget_s>
"""
import os, sys, time, re, numpy as np, onnx, onnxruntime as ort
from onnx import helper, TensorProto
from PIL import Image, ImageOps
root, work, budget = sys.argv[1], sys.argv[2], float(sys.argv[3]); t0 = time.time()
m = onnx.load(os.path.expanduser('~/models/mobilenetv2-12.onnx'))
m.graph.output.extend([helper.make_tensor_value_info('472', TensorProto.FLOAT, None)])
so = ort.SessionOptions(); so.intra_op_num_threads = os.cpu_count()
s = ort.InferenceSession(m.SerializeToString(), so, providers=['CPUExecutionProvider'])
mean = np.array([0.485, 0.456, 0.406], np.float32); std = np.array([0.229, 0.224, 0.225], np.float32)
num = lambda f: int(re.search(r'_(\d{5})\.jpg$', f).group(1))
for c in sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))], key=str.lower):
    out = os.path.join(work, 'feat', c + '.npz')
    if os.path.exists(out): continue
    if time.time() - t0 > budget: print('stopping (time budget)'); break
    files = sorted([f for f in os.listdir(os.path.join(root, c)) if f.lower().endswith('.jpg')], key=num)
    os.makedirs(os.path.join(work, 'thumbs', c), exist_ok=True)
    X = []
    for f in files:
        im = Image.open(os.path.join(root, c, f)); im.draft('RGB', (448, 448))
        im = ImageOps.exif_transpose(im).convert('RGB')
        th = ImageOps.fit(im, (150, 150)); th.save(os.path.join(work, 'thumbs', c, f), quality=80)
        X.append(((np.asarray(ImageOps.fit(im, (224, 224)), np.float32) / 255. - mean) / std).transpose(2, 0, 1))
    F = np.concatenate([s.run(['472'], {s.get_inputs()[0].name: np.stack(X[i:i+32])})[0].reshape(-1, 1280) for i in range(0, len(X), 32)])
    np.savez(out, files=np.array(files), F=F.astype(np.float32))
    print(c, len(files), f'{time.time()-t0:.0f}s', flush=True)
