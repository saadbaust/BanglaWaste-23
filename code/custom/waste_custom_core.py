# ===================== CELL: SETUP =====================
import os, sys, json, time, math, random, glob, shutil, subprocess, warnings
warnings.filterwarnings("ignore")
try:
    import timm
    from packaging.version import Version
    if Version(timm.__version__) < Version("1.0.9"):
        raise ImportError
except Exception:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "timm"], check=True)
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import timm
from timm.data import Mixup, resolve_data_config
from timm.loss import SoftTargetCrossEntropy
from timm.utils import ModelEmaV3
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, precision_recall_fscore_support

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("torch", torch.__version__, "| timm", timm.__version__, "| device", DEVICE,
      "|", torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU")

# ImageNet-1k weights for every model (same pretraining data for a fair comparison)
MODEL_TAGS = {
    "mobilenetv2_100":        "mobilenetv2_100.ra_in1k",
    "mobilenetv3_large_100":  "mobilenetv3_large_100.ra_in1k",
    "mobilenetv4_conv_small": "mobilenetv4_conv_small.e2400_r224_in1k",
    "efficientnet_b0":        "efficientnet_b0.ra_in1k",
    "ghostnetv2_100":         "ghostnetv2_100.in1k",
    "mobilevitv2_100":        "mobilevitv2_100.cvnets_in1k",
    "efficientformerv2_s0":   "efficientformerv2_s0.snap_dist_in1k",
    "resnet50":               "resnet50.a1_in1k",
    "densenet121":            "densenet121.ra_in1k",
    "convnext_tiny":          "convnext_tiny.fb_in1k",
    "deit_small_patch16_224": "deit_small_patch16_224.fb_in1k",
}

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True

# ===================== CELL: DATA =====================
def find_data():
    hits = glob.glob(os.path.join(DATA_ROOT_SEARCH, "**", "folds.csv"), recursive=True)
    if not hits:
        raise FileNotFoundError(f"folds.csv not found under {DATA_ROOT_SEARCH}. Did you add the dataset (Add Input)?")
    split_dir = os.path.dirname(hits[0])
    imgs = glob.glob(os.path.join(DATA_ROOT_SEARCH, "**", "BanglaWaste23_256"), recursive=True)
    imgs = [p for p in imgs if os.path.isdir(p)]
    if not imgs:
        raise FileNotFoundError("BanglaWaste23_256 folder not found in the input dataset.")
    return split_dir, imgs[0]

SPLIT_DIR, IMG_DIR = find_data()
FOLDS = pd.read_csv(os.path.join(SPLIT_DIR, "folds.csv"))
CLASS_MAP = pd.read_csv(os.path.join(SPLIT_DIR, "class_map.csv"))
FOLD0_VAL = pd.read_csv(os.path.join(SPLIT_DIR, "fold0_val.csv"))
RANDOM_SPLIT = pd.read_csv(os.path.join(SPLIT_DIR, "random_split.csv"))
NUM_CLASSES = len(CLASS_MAP)
CLASS_NAMES = CLASS_MAP.class_folder.tolist()
assert len(FOLDS) == 4160 and NUM_CLASSES == 23
print("splits:", SPLIT_DIR, "| images:", IMG_DIR)

# ---- Data loading: RAM cache (same idea as TensorFlow's .cache()) ----
# CACHE_IN_RAM = True : every image is decoded ONCE into memory (about 1 GB for 4,160 images at 256 px);
#                       each epoch then only does augmentation -> much faster than reading JPEGs every epoch.
# CACHE_IN_RAM = False: images are read from disk every time they are used (use only if RAM runs out).
# Augmentation is applied AFTER the cache, so every epoch still sees new random crops/flips.
t0 = time.time()
KEEP = set()
for c, g in FOLDS.groupby("class_folder"):
    names = g.file_name.tolist()[:LIMIT_IMAGES_PER_CLASS] if LIMIT_IMAGES_PER_CLASS else g.file_name.tolist()
    KEEP.update(f"{c}/{n}" for n in names)
CACHE = {}
if CACHE_IN_RAM:
    for k in sorted(KEEP):
        with Image.open(os.path.join(IMG_DIR, k)) as im:
            CACHE[k] = np.asarray(im.convert("RGB")).copy()
    ram_gb = sum(a.nbytes for a in CACHE.values()) / 1e9
    print(f"cached {len(CACHE)} images in RAM ({ram_gb:.2f} GB) in {time.time() - t0:.0f} s")
else:
    print(f"RAM cache OFF: {len(KEEP)} images will be read from disk every epoch")

def load_image(k):
    if CACHE_IN_RAM:
        return Image.fromarray(CACHE[k])
    with Image.open(os.path.join(IMG_DIR, k)) as im:
        return im.convert("RGB")

def split_frames(job):
    """Return (train_df, eval_df, eval_name) for a job. Test folds are only used in mode 'cv'."""
    key = lambda d: d.class_folder + "/" + d.file_name
    if job["split"] == "fold0_val":
        tr = FOLD0_VAL[~FOLD0_VAL.is_val]; ev = FOLD0_VAL[FOLD0_VAL.is_val]; name = "val"
    elif job["split"] == "cv":
        k = job["fold"]; tr = FOLDS[FOLDS.fold != k]; ev = FOLDS[FOLDS.fold == k]; name = "test"
    elif job["split"] == "random":
        tr = RANDOM_SPLIT[RANDOM_SPLIT.split == "train"]; ev = RANDOM_SPLIT[RANDOM_SPLIT.split == "test"]; name = "test"
    else:
        raise ValueError(job["split"])
    tr = tr[key(tr).isin(KEEP)]; ev = ev[key(ev).isin(KEEP)]
    return tr.reset_index(drop=True), ev.reset_index(drop=True), name

class WasteDS(Dataset):
    def __init__(self, df, tf):
        self.keys = (df.class_folder + "/" + df.file_name).tolist(); self.y = df.class_id.tolist(); self.tf = tf
    def __len__(self): return len(self.keys)
    def __getitem__(self, i):
        return self.tf(load_image(self.keys[i])), self.y[i]

def make_tfs(mean, std):
    train = T.Compose([
        T.RandomResizedCrop(224, scale=(0.6, 1.0)),
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
        T.RandomRotation(20),
        T.ColorJitter(brightness=0.15, contrast=0.15),
        T.ToTensor(), T.Normalize(mean, std)])
    evalt = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean, std)])
    return train, evalt

# ===================== CELL: CUSTOM_MODELS =====================
# Custom models.
#   WasteMSANet  = PyTorch port of the original Keras "Waste_Classification_Model" (ERB + multi-scale feature
#                  module (MFM) + channel/spatial attention + dual-pool head). V0 = exact original layout.
#   HybridMSA    = ImageNet-pretrained timm backbone + the same MFM/attention ideas as a neck (Path B).
class ChannelAttention(nn.Module):            # CBAM-style: shared MLP on global avg + max
    def __init__(self, ch, ratio=16):
        super().__init__(); h = max(ch // ratio, 4)
        self.fc1 = nn.Linear(ch, h, bias=False); self.fc2 = nn.Linear(h, ch, bias=False)
    def forward(self, x):
        a = self.fc2(F.relu(self.fc1(x.mean((2, 3))))); m = self.fc2(F.relu(self.fc1(x.amax((2, 3)))))
        return x * torch.sigmoid(a + m)[:, :, None, None]

class SpatialAttention(nn.Module):            # as in the Keras model: 7x7 conv on the feature map -> 1 channel mask
    def __init__(self, ch):
        super().__init__(); self.conv = nn.Conv2d(ch, 1, 7, padding=3, bias=False)
    def forward(self, x): return x * torch.sigmoid(self.conv(x))

class ERB(nn.Module):                         # enhanced residual block
    def __init__(self, cin, cout, groups=1, stride=1, attention=True, projection=False):
        super().__init__(); b = cout // 2
        self.body = nn.Sequential(
            nn.Conv2d(cin, b, 1, bias=True), nn.BatchNorm2d(b), nn.SiLU(),
            nn.Conv2d(b, b, 3, stride, 1, groups=groups, bias=True), nn.BatchNorm2d(b), nn.SiLU(),
            nn.Conv2d(b, cout, 1, bias=True), nn.BatchNorm2d(cout))
        self.att = nn.Sequential(ChannelAttention(cout), SpatialAttention(cout)) if attention else nn.Identity()
        self.short = (nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=True), nn.BatchNorm2d(cout))
                      if (projection or stride > 1 or cin != cout) else nn.Identity())
        self.act = nn.SiLU()
    def forward(self, x): return self.act(self.att(self.body(x)) + self.short(x))

class MFM(nn.Module):                         # multi-scale feature module: DW 3/5/7 -> fuse -> SE -> residual
    def __init__(self, cin, c, drop=0.1, kernels=(3, 5, 7), se=True):
        super().__init__()
        self.short = nn.Sequential(nn.Conv2d(cin, c, 1), nn.BatchNorm2d(c))
        self.pre = nn.Sequential(nn.Conv2d(cin, c, 1), nn.BatchNorm2d(c), nn.GELU())
        self.branches = nn.ModuleList([nn.Sequential(nn.Conv2d(c, c, k, padding=k // 2, groups=c), nn.BatchNorm2d(c), nn.GELU())
                                       for k in kernels])
        self.fuse = nn.Sequential(nn.Conv2d(c * len(kernels), c, 1), nn.BatchNorm2d(c), nn.GELU())
        h = max(c // 8, 4)
        self.se = nn.Sequential(nn.Linear(c, h, bias=False), nn.ReLU(), nn.Linear(h, c, bias=False), nn.Sigmoid()) if se else None
        self.drop = nn.Dropout2d(drop)
    def forward(self, x):
        s = self.short(x); x = self.pre(x)
        x = self.fuse(torch.cat([b(x) for b in self.branches], 1))
        if self.se is not None: x = x * self.se(x.mean((2, 3)))[:, :, None, None]
        return self.drop(x + s)

class DualPoolHead(nn.Module):                # concat(GMP, GAP) -> 128 -> 64 -> classes (as in the Keras model)
    def __init__(self, c, num_classes):
        super().__init__()
        self.mlp = nn.Sequential(nn.Dropout(0.4), nn.Linear(2 * c, 128), nn.SiLU(), nn.BatchNorm1d(128), nn.Dropout(0.3),
                                 nn.Linear(128, 64), nn.SiLU(), nn.BatchNorm1d(64), nn.Linear(64, num_classes))
    def forward(self, x): return self.mlp(torch.cat([x.amax((2, 3)), x.mean((2, 3))], 1))

# stage spec: (filters, groups, n_blocks, attention, mfm, mfm_dropout, pool_after)
CUSTOM_SPECS = {
    # V0 = original Keras model, 4 stages, final map 14x14
    "custom_v0": [(32, 2, 2, True, True, 0.05, True), (64, 4, 2, True, True, 0.10, True),
                  (96, 8, 2, True, True, 0.15, False), (128, 16, 2, True, True, 0.20, False)],
    # V1 = redesign from the old ablation: light early stages (1 block, no attention, no MFM, only downsampling),
    #      attention + MFM only in the later stages, extra downsampling so the last stage works at 7x7, wider end
    "custom_v1": [(32, 2, 1, False, False, 0.0, True), (64, 4, 1, False, False, 0.0, True),
                  (96, 8, 2, True, True, 0.15, True), (160, 16, 2, True, True, 0.20, False)],
}

class WasteMSANet(nn.Module):
    def __init__(self, spec, num_classes):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 5, 2, 2), nn.BatchNorm2d(32), nn.SiLU(), nn.MaxPool2d(3, 2, 1))
        layers, cin = [], 32
        for f, g, n, att, mfm, drop, pool in spec:
            for i in range(n):
                layers.append(ERB(cin, f, groups=g, attention=att, projection=(i == 0))); cin = f
            if mfm: layers.append(MFM(cin, f, drop))
            if pool: layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)
        self.head = DualPoolHead(cin, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    def forward(self, x): return self.head(self.features(self.stem(x)))

class HybridMSA(nn.Module):                   # Path B: pretrained backbone + MFM/attention neck + dual-pool head
    def __init__(self, backbone_tag, num_classes, neck=160, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(backbone_tag, pretrained=pretrained, features_only=True, out_indices=(-1,))
        cin = self.backbone.feature_info.channels()[-1]
        self.reduce = nn.Sequential(nn.Conv2d(cin, neck, 1, bias=False), nn.BatchNorm2d(neck), nn.SiLU())
        self.mfm = MFM(neck, neck, 0.2)
        self.att = nn.Sequential(ChannelAttention(neck), SpatialAttention(neck))
        self.head = DualPoolHead(neck, num_classes)
    def forward(self, x):
        x = self.reduce(self.backbone(x)[-1])
        return self.head(self.att(self.mfm(x)))

HYBRID_BACKBONES = {"hyb_mnv3_msa": "mobilenetv3_large_100.ra_in1k"}

def build_model(name):
    """Returns (model, mean, std). Custom models use ImageNet mean/std like every other model."""
    if name in CUSTOM_SPECS:
        return WasteMSANet(CUSTOM_SPECS[name], NUM_CLASSES), IMAGENET_MEAN, IMAGENET_STD
    if name in HYBRID_BACKBONES:
        m = HybridMSA(HYBRID_BACKBONES[name], NUM_CLASSES, pretrained=PRETRAINED)
        return m, IMAGENET_MEAN, IMAGENET_STD
    m = timm.create_model(MODEL_TAGS[name], pretrained=PRETRAINED, num_classes=NUM_CLASSES)
    cfg = resolve_data_config({}, model=m)
    return m, cfg["mean"], cfg["std"]

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

# ===================== CELL: TRAINING =====================
def count_macs(model):
    try:
        from torch.utils.flop_counter import FlopCounterMode
        m = model.eval(); x = torch.zeros(1, 3, 224, 224, device=next(m.parameters()).device)
        with FlopCounterMode(display=False) as fc, torch.no_grad():
            m(x)
        return fc.get_total_flops() / 2 / 1e6
    except Exception as e:
        print("MAC count failed:", e); return float("nan")

@torch.no_grad()
def predict(model, loader):
    model.eval(); P = []
    for x, _ in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
            P.append(torch.softmax(model(x.to(DEVICE, non_blocking=True)).float(), 1).cpu())
    return torch.cat(P).numpy()

def metrics_from(y, p):
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=range(NUM_CLASSES), zero_division=0)
    return {"accuracy": accuracy_score(y, p), "macro_f1": f1_score(y, p, average="macro", labels=range(NUM_CLASSES), zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y, p), "weighted_f1": f1_score(y, p, average="weighted", zero_division=0),
            "per_class": {CLASS_NAMES[i]: {"precision": float(pr[i]), "recall": float(rc[i]), "f1": float(f1[i]), "support": int(sup[i])}
                          for i in range(NUM_CLASSES)}}

def run_job(job, return_weights_path=False):
    run_name = f"{job['model']}_lr{job['lr']:g}" + ("_kd" if job.get("kd") else "") + ("_teacher" if return_weights_path else "")
    sub = {"fold0_val": "fold0val", "cv": f"fold{job.get('fold')}", "random": "random"}[job["split"]]
    out = os.path.join(OUT_DIR, job["split"], run_name, sub)
    wpath = os.path.join(out, "final_ema_fp16.pt")
    if os.path.exists(os.path.join(out, "metrics.json")) and (not return_weights_path or os.path.exists(wpath)):
        print("SKIP (already done):", out); return wpath
    os.makedirs(out, exist_ok=True)
    seed = SEED + (job.get("fold") or 0); seed_all(seed)

    model, mean, std = build_model(job["model"]); model = model.to(DEVICE)
    cfg = {"mean": mean, "std": std}
    tf_train, tf_eval = make_tfs(mean, std)
    teacher = None
    if job.get("kd"):                                              # knowledge distillation from a teacher
        kd = job["kd"]                                             # trained on THIS job's training data only
        tjob = {"model": kd["teacher"], "lr": kd["teacher_lr"], "epochs": kd["teacher_epochs"],
                "split": job["split"], "fold": job.get("fold"), "pretrained_override": True}
        tpath = run_job(tjob, return_weights_path=True)
        teacher, _, _ = build_model(kd["teacher"])
        teacher.load_state_dict({k: v.float() for k, v in torch.load(tpath, map_location="cpu").items()})
        teacher = teacher.to(DEVICE).eval()
        for q in teacher.parameters(): q.requires_grad_(False)
        seed_all(seed)
    tr, ev, ev_name = split_frames(job)
    g = torch.Generator(); g.manual_seed(seed)
    dl_tr = DataLoader(WasteDS(tr, tf_train), batch_size=BATCH, shuffle=True, drop_last=True,
                       num_workers=WORKERS, pin_memory=True, generator=g, persistent_workers=WORKERS > 0)
    dl_ev = DataLoader(WasteDS(ev, tf_eval), batch_size=64, shuffle=False, num_workers=WORKERS, pin_memory=True)

    params = sum(p.numel() for p in model.parameters()); macs = count_macs(model); model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=job["lr"], weight_decay=WEIGHT_DECAY)
    steps_per_ep = len(dl_tr); total = job["epochs"] * steps_per_ep; warm = WARMUP_EPOCHS * steps_per_ep
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm
                                              else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    ema = ModelEmaV3(model, decay=EMA_DECAY, use_warmup=True)
    mix = Mixup(mixup_alpha=0.2, cutmix_alpha=1.0, prob=0.5, switch_prob=0.5, label_smoothing=0.0, num_classes=NUM_CLASSES)
    crit = SoftTargetCrossEntropy(); scaler = torch.amp.GradScaler(enabled=DEVICE == "cuda")
    y_ev = ev.class_id.values

    hist = []; t0 = time.time(); step = 0
    print(f"\n=== {run_name} | {sub} | train {len(tr)} | {ev_name} {len(ev)} | params {params/1e6:.2f} M | MACs {macs:.0f} M")
    for ep in range(1, job["epochs"] + 1):
        model.train(); tot = 0.0
        for x, y in dl_tr:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            x, y = mix(x, y)
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
                logits = model(x); loss = crit(logits, y)
                if teacher is not None:
                    with torch.no_grad():
                        tl = teacher(x)
                    Tk, ak = job["kd"]["T"], job["kd"]["alpha"]
                    kl = F.kl_div(F.log_softmax(logits.float() / Tk, 1), F.softmax(tl.float() / Tk, 1), reduction="batchmean")
                    loss = (1 - ak) * loss + ak * Tk * Tk * kl
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step(); ema.update(model, step=step); step += 1
            tot += loss.item()
        row = {"epoch": ep, "train_loss": tot / steps_per_ep, "lr": opt.param_groups[0]["lr"], "minutes": (time.time() - t0) / 60}
        # validation curve ONLY on the fold-0 inner validation set (never on test folds)
        if job["split"] == "fold0_val" and (ep % VAL_EVERY == 0 or ep == job["epochs"]):
            pe = predict(ema.module, dl_ev).argmax(1); m = metrics_from(y_ev, pe)
            row.update({"val_acc_ema": m["accuracy"], "val_macro_f1_ema": m["macro_f1"]})
        hist.append(row)
        if ep % PRINT_EVERY == 0 or ep == job["epochs"]:
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})

    P = predict(ema.module, dl_ev); pred = P.argmax(1); m = metrics_from(y_ev, pred)
    t_inf = time.time(); predict(ema.module, dl_ev); infer_s = time.time() - t_inf
    # --- everything needed later for tables and figures, so this run never has to be repeated ---
    pd.DataFrame(hist).to_csv(os.path.join(out, "history.csv"), index=False)                      # training curves
    pr = pd.DataFrame({"file_name": ev.file_name, "class_folder": ev.class_folder, "object_id": ev.object_id,
                       "y_true": y_ev, "y_pred": pred, "confidence": P.max(1)})
    pr = pd.concat([pr, pd.DataFrame(P, columns=[f"p_{c}" for c in CLASS_NAMES])], axis=1)
    pr.to_csv(os.path.join(out, "predictions.csv"), index=False)                                   # per image + all probabilities
    cm = pd.crosstab(pd.Categorical(y_ev, range(NUM_CLASSES)), pd.Categorical(pred, range(NUM_CLASSES)), dropna=False)
    cm.index = CLASS_NAMES; cm.columns = CLASS_NAMES
    cm.to_csv(os.path.join(out, "confusion_matrix.csv"))                                           # counts, rows = true
    pd.DataFrame(m["per_class"]).T.rename_axis("class").to_csv(os.path.join(out, "per_class.csv"))  # precision/recall/F1/support
    info = {"run": run_name, "split": job["split"], "fold": job.get("fold"), "eval_set": ev_name, "model": job["model"],
            "timm_tag": MODEL_TAGS.get(job["model"], job["model"]),
            "pretrained": PRETRAINED and job["model"] not in CUSTOM_SPECS, "kd": json.dumps(job.get("kd")) if job.get("kd") else "", "lr": job["lr"], "epochs": job["epochs"],
            "batch": BATCH, "weight_decay": WEIGHT_DECAY, "warmup_epochs": WARMUP_EPOCHS, "ema_decay": EMA_DECAY,
            "seed": seed, "params": params, "macs_M": macs, "train_images": len(tr), "eval_images": len(ev),
            "minutes": (time.time() - t0) / 60, "sec_per_epoch": (time.time() - t0) / job["epochs"],
            "gpu_infer_img_per_s": len(ev) / max(infer_s, 1e-9), "cache_in_ram": CACHE_IN_RAM,
            "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu",
            "torch": torch.__version__, "timm": timm.__version__, "mean": list(cfg["mean"]), "std": list(cfg["std"]),
            "slot": SLOT, **{k: v for k, v in m.items() if k != "per_class"}}
    if job["split"] == "fold0_val":                                                              # best point of the val curve
        h = pd.DataFrame(hist).dropna(subset=["val_macro_f1_ema"])
        b = h.loc[h.val_macro_f1_ema.idxmax()]
        info.update({"best_val_macro_f1": float(b.val_macro_f1_ema), "best_val_epoch": int(b.epoch)})
    json.dump(info, open(os.path.join(out, "metrics.json"), "w"), indent=2)
    pd.DataFrame([{k: (str(v) if isinstance(v, list) else v) for k, v in info.items()}]).to_csv(
        os.path.join(out, "metrics.csv"), index=False)                                             # one row, easy to stack
    json.dump(job, open(os.path.join(out, "job.json"), "w"), indent=2)
    if job["model"] in SAVE_WEIGHTS_FOR or return_weights_path:
        torch.save({k: v.half() for k, v in ema.module.state_dict().items()}, os.path.join(out, "final_ema_fp16.pt"))
    print(f"DONE {run_name} {sub}: acc {m['accuracy']:.4f} | macro-F1 {m['macro_f1']:.4f} | {info['minutes']:.1f} min")
    del model, ema, opt, teacher; torch.cuda.empty_cache()
    return wpath

# ===================== CELL: RUN =====================
os.makedirs(OUT_DIR, exist_ok=True)
for j in JOBS:
    run_job(j)

# ===================== CELL: PACK =====================
mfiles = sorted(glob.glob(os.path.join(OUT_DIR, "**", "metrics.csv"), recursive=True))
summary = pd.concat([pd.read_csv(f) for f in mfiles], ignore_index=True) if mfiles else pd.DataFrame()
summary.to_csv(os.path.join(OUT_DIR, f"summary_{SLOT}.csv"), index=False)                       # all runs of this notebook
cols = [c for c in ["split", "run", "fold", "eval_set", "accuracy", "macro_f1", "best_val_macro_f1", "best_val_epoch",
                    "params", "macs_M", "minutes"] if c in summary.columns]
print(summary[cols].to_string(index=False) if len(summary) else "no finished runs")
env = {"slot": SLOT, "jobs": JOBS, "torch": torch.__version__, "timm": timm.__version__, "python": sys.version,
       "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu", "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
json.dump(env, open(os.path.join(OUT_DIR, f"environment_{SLOT}.json"), "w"), indent=2, default=str)
import zipfile
zip_path = os.path.join(os.path.dirname(OUT_DIR), f"results_{SLOT}.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:   # results only; model weights stay in the Output tab
    for f in glob.glob(os.path.join(OUT_DIR, "**", "*"), recursive=True):
        if os.path.isfile(f) and not f.endswith(".pt"):
            zf.write(f, os.path.relpath(f, os.path.dirname(OUT_DIR)))
print("Download this file from the Output tab:", zip_path)
