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
from PIL import Image, ImageOps
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
CACHE = {}; BDW_CACHE = {}
if CACHE_IN_RAM:
    for k in sorted(KEEP):
        with Image.open(os.path.join(IMG_DIR, k)) as im:
            CACHE[k] = np.asarray(im.convert("RGB")).copy()
    ram_gb = sum(a.nbytes for a in CACHE.values()) / 1e9
    print(f"cached {len(CACHE)} images in RAM ({ram_gb:.2f} GB) in {time.time() - t0:.0f} s")
else:
    print(f"RAM cache OFF: {len(KEEP)} images will be read from disk every epoch")

def load_image(k):
    if k in BDW_CACHE:
        return Image.fromarray(BDW_CACHE[k])
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
    elif job["split"] == "all_bdwaste":                            # train on ALL our images, test on BDWaste
        tr = FOLDS[key(FOLDS).isin(KEEP)]; return tr.reset_index(drop=True), BDW.copy(), "bdwaste"
    elif job["split"] == "random":
        tr = RANDOM_SPLIT[RANDOM_SPLIT.split == "train"]; ev = RANDOM_SPLIT[RANDOM_SPLIT.split == "test"]; name = "test"
    else:
        raise ValueError(job["split"])
    tr = tr[key(tr).isin(KEEP)]; ev = ev[key(ev).isin(KEEP)]
    return tr.reset_index(drop=True), ev.reset_index(drop=True), name

class WasteDS(Dataset):
    def __init__(self, df, tf):
        self.keys = (df["key"] if "key" in df.columns else df.class_folder + "/" + df.file_name).tolist()
        self.y = df.class_id.tolist(); self.tf = tf
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
    if name.endswith("_scratch"):                          # same architecture, random initial weights (no ImageNet)
        m = timm.create_model(MODEL_TAGS[name[:-len("_scratch")]], pretrained=False, num_classes=NUM_CLASSES)
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
    sub = {"fold0_val": "fold0val", "cv": f"fold{job.get('fold')}", "random": "random", "all_bdwaste": "all"}[job["split"]]
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
            "timm_tag": MODEL_TAGS.get(job["model"].replace("_scratch", ""), job["model"]),
            "pretrained": PRETRAINED and job["model"] not in CUSTOM_SPECS and not job["model"].endswith("_scratch"), "kd": json.dumps(job.get("kd")) if job.get("kd") else "", "lr": job["lr"], "epochs": job["epochs"],
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
    if job["split"] == "cv" and (job["model"], job.get("fold")) in GRADCAM_AFTER:
        gradcam_report(ema.module, mean, std, ev, os.path.join(out, "gradcam"), f"{job['model']} fold {job.get('fold')}")
    if job["model"] in SAVE_WEIGHTS_FOR or return_weights_path:
        torch.save({k: v.half() for k, v in ema.module.state_dict().items()}, os.path.join(out, "final_ema_fp16.pt"))
    print(f"DONE {run_name} {sub}: acc {m['accuracy']:.4f} | macro-F1 {m['macro_f1']:.4f} | {info['minutes']:.1f} min")
    del model, ema, opt, teacher; torch.cuda.empty_cache()
    return wpath

# ===================== CELL: EXTRAS =====================
# Helpers for batch 3: reuse saved weights, Grad-CAM, shortcut (re-encoding) test, BDWaste cross-dataset, TFLite export.
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def run_dir(job):
    name = f"{job['model']}_lr{job['lr']:g}" + ("_kd" if job.get("kd") else "")
    sub = {"fold0_val": "fold0val", "cv": f"fold{job.get('fold')}", "random": "random", "all_bdwaste": "all"}[job["split"]]
    return name, sub

def find_saved_weights(job):
    """Look for final_ema_fp16.pt of this job in any added input (e.g. the Round 2 notebook outputs)."""
    name, sub = run_dir(job)
    hits = glob.glob(os.path.join("/kaggle/input", "**", job["split"], name, sub, "final_ema_fp16.pt"), recursive=True) \
        + glob.glob(os.path.join(OUT_DIR, job["split"], name, sub, "final_ema_fp16.pt"))
    return hits[0] if hits else None

def get_trained(job):
    """Return (model on DEVICE in eval mode, mean, std). Loads saved weights if found, else trains the job now."""
    w = find_saved_weights(job)
    if w is None:
        print(f"no saved weights for {run_dir(job)} -> training now")
        job = dict(job); SAVE_WEIGHTS_FOR.add(job["model"]); run_job(job)
        w = find_saved_weights(job)
    else:
        print("using saved weights:", w)
    m, mean, std = build_model(job["model"])
    m.load_state_dict({k: v.float() for k, v in torch.load(w, map_location="cpu").items()})
    return m.to(DEVICE).eval(), mean, std

def eval_frame_for(job):
    return split_frames(job)[1]

# ---------- Grad-CAM (computed on the last 4-D feature map before global pooling)
def feats_and_logits(model, x):
    if isinstance(model, WasteMSANet):
        f = model.features(model.stem(x)); return f, model.head(f)
    if isinstance(model, HybridMSA):
        f = model.att(model.mfm(model.reduce(model.backbone(x)[-1]))); return f, model.head(f)
    f = model.forward_features(x)
    return f, model.forward_head(f)

def gradcam_report(model, mean, std, ev, out_dir, title, n_wrong=8):
    """Saves gradcam_correct.png/pdf (best correct image per class), gradcam_wrong.png/pdf (most confident errors)
    and gradcam_data.npz (images, CAMs, labels) so the figures can be redrawn without the GPU."""
    os.makedirs(out_dir, exist_ok=True)
    _, tf_eval = make_tfs(mean, std)
    keys = (ev.class_folder + "/" + ev.file_name).tolist(); y = ev.class_id.values
    model.eval(); P = []
    with torch.no_grad():
        for i in range(0, len(keys), 64):
            xb = torch.stack([tf_eval(load_image(k)) for k in keys[i:i + 64]]).to(DEVICE)
            P.append(torch.softmax(model(xb).float(), 1).cpu().numpy())
    P = np.concatenate(P); pred = P.argmax(1); conf = P.max(1)
    sel = []
    for c in range(NUM_CLASSES):
        idx = np.flatnonzero((y == c) & (pred == c))
        if len(idx): sel.append(("correct", int(idx[np.argmax(conf[idx])])))
    wrong = np.flatnonzero(pred != y); wrong = wrong[np.argsort(-conf[wrong])][:n_wrong]
    sel += [("wrong", int(i)) for i in wrong]
    imgs, cams, rows = [], [], []
    m_ = torch.tensor(mean).view(3, 1, 1); s_ = torch.tensor(std).view(3, 1, 1)
    for kind, i in sel:
        x = tf_eval(load_image(keys[i])).unsqueeze(0).to(DEVICE).requires_grad_(False)
        with torch.enable_grad():
            f, logits = feats_and_logits(model, x); f.retain_grad()
            logits[0, pred[i]].backward()
            w = f.grad.mean((2, 3), keepdim=True); cam = F.relu((w * f).sum(1))[0]
        cam = (cam / (cam.max() + 1e-8)).detach().cpu().numpy()
        img = (x[0].cpu() * s_ + m_).clamp(0, 1).permute(1, 2, 0).numpy()
        imgs.append((img * 255).astype(np.uint8)); cams.append(cam)
        rows.append({"kind": kind, "file_name": ev.file_name.iloc[i], "object_id": ev.object_id.iloc[i],
                     "true": CLASS_NAMES[y[i]], "pred": CLASS_NAMES[pred[i]], "confidence": float(conf[i])})
        model.zero_grad(set_to_none=True)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "gradcam_items.csv"), index=False)
    np.savez_compressed(os.path.join(out_dir, "gradcam_data.npz"), images=np.stack(imgs),
                        cams=np.stack([np.asarray(c, dtype=np.float32) for c in cams]) if len({c.shape for c in cams}) == 1 else np.array(cams, dtype=object))
    for kind in ("correct", "wrong"):
        items = [(r, im, cm) for r, im, cm in zip(rows, imgs, cams) if r["kind"] == kind]
        if not items: continue
        cols = 6 if kind == "correct" else 4; nrow = math.ceil(len(items) / cols)
        fig, axes = plt.subplots(nrow * 2, cols, figsize=(2.2 * cols, 4.6 * nrow))
        axes = np.array(axes).reshape(nrow * 2, cols)
        for a in axes.ravel(): a.axis("off")
        for j, (r, im, cm) in enumerate(items):
            rr, cc = 2 * (j // cols), j % cols
            axes[rr, cc].imshow(im)
            axes[rr, cc].set_title(r["true"][:18] if kind == "correct" else f"T: {r['true'][:14]}\nP: {r['pred'][:14]}", fontsize=7)
            up = torch.nn.functional.interpolate(torch.tensor(cm)[None, None], size=im.shape[:2], mode="bilinear")[0, 0].numpy()
            axes[rr + 1, cc].imshow(im); axes[rr + 1, cc].imshow(up, cmap="jet", alpha=0.45)
        fig.suptitle(f"Grad-CAM — {title} — {kind}", fontsize=10); fig.tight_layout()
        for ext in ("png", "pdf"): fig.savefig(os.path.join(out_dir, f"gradcam_{kind}.{ext}"), dpi=150)
        plt.close(fig)
    print(f"Grad-CAM saved: {out_dir} ({len(rows)} images)")

# ---------- Shortcut test: same test images, degraded, same model
def degrade(img, kind):
    if kind == "original": return img
    if kind == "jpeg_q30":
        b = io.BytesIO(); img.save(b, "JPEG", quality=30); b.seek(0); return Image.open(b).convert("RGB")
    if kind == "lowres_96":
        w, h = img.size; s = 96 / min(w, h)
        return img.resize((round(w * s), round(h * s)), Image.BILINEAR).resize((w, h), Image.BICUBIC)
    raise ValueError(kind)

def shortcut_eval(model, mean, std, ev, out_dir, conditions=("original", "jpeg_q30", "lowres_96")):
    os.makedirs(out_dir, exist_ok=True); _, tf_eval = make_tfs(mean, std)
    keys = (ev.class_folder + "/" + ev.file_name).tolist(); y = ev.class_id.values; rows = []
    malta = CLASS_NAMES.index("malta_peeling")
    for cond in conditions:
        P = []
        with torch.no_grad():
            for i in range(0, len(keys), 64):
                xb = torch.stack([tf_eval(degrade(load_image(k), cond)) for k in keys[i:i + 64]]).to(DEVICE)
                with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
                    P.append(torch.softmax(model(xb).float(), 1).cpu().numpy())
        P = np.concatenate(P); pred = P.argmax(1)
        pd.DataFrame({"file_name": ev.file_name, "object_id": ev.object_id, "y_true": y, "y_pred": pred,
                      "confidence": P.max(1)}).to_csv(os.path.join(out_dir, f"predictions_{cond}.csv"), index=False)
        rows.append({"condition": cond, "accuracy": accuracy_score(y, pred),
                     "macro_f1": f1_score(y, pred, average="macro", labels=range(NUM_CLASSES), zero_division=0),
                     "pred_as_malta_not_malta": int(((pred == malta) & (y != malta)).sum()),
                     "malta_recall": float((pred[y == malta] == malta).mean()) if (y == malta).any() else float("nan")})
    r = pd.DataFrame(rows); r.to_csv(os.path.join(out_dir, "shortcut_summary.csv"), index=False); print(r.round(4).to_string(index=False))
    return r

# ===================== CELL: BDWASTE_AND_EXPORT =====================
# ---------- BDWaste (cross-dataset test). Shared classes are matched by keywords in BDWaste folder names.
BDW_KEYWORDS = {                      # checked in this order; the first match wins
    "medicine": ["medicine", "tablet", "pill"],
    "Banana_peeling": ["banana"], "Bottle": ["bottle"], "Egg": ["egg"], "Lemon": ["lemon"],
    "malta_peeling": ["malta"], "mango_peeling": ["mango"],
    "packet": ["packet", "chips", "wrapper", "snack"], "polythene": ["polythene", "polybag", "poly bag", "plastic bag", "shopping bag"],
    "potato_peeling": ["potato"], "sugarcane": ["sugarcane", "sugar cane"],
}
BDW = None

def build_bdwaste():
    """Find BDWaste images (folders under Digestible/Indigestible), map sub-folders to our classes, load into RAM."""
    global BDW
    imgs = [p for p in glob.glob(os.path.join(DATA_ROOT_SEARCH, "**", "*"), recursive=True)
            if p.lower().endswith((".jpg", ".jpeg", ".png")) and "igestible" in p.lower()]
    if not imgs:
        raise FileNotFoundError("No BDWaste images found (folders containing 'Digestible'/'Indigestible'). Add the BDWaste dataset as input.")
    df = pd.DataFrame({"path": imgs}); df["subfolder"] = df.path.map(lambda p: os.path.basename(os.path.dirname(p)))
    df["top"] = df.path.map(lambda p: "Indigestible" if "indigestible" in p.lower() else "Digestible")
    def match(name):
        n = name.lower().replace("_", " ").replace("-", " ")
        hits = [c for c, kws in BDW_KEYWORDS.items() if any(k in n for k in kws)]
        if name in BDW_OVERRIDE: return BDW_OVERRIDE[name]
        return hits[0] if hits else ""
    table = df.groupby(["top", "subfolder"]).size().rename("images").reset_index()
    table["our_class"] = table.subfolder.map(match)
    os.makedirs(OUT_DIR, exist_ok=True); table.to_csv(os.path.join(OUT_DIR, "bdwaste_mapping.csv"), index=False)
    print("BDWaste folders and mapping (edit BDW_OVERRIDE in CONFIG if a match is wrong):"); print(table.to_string(index=False))
    df["our_class"] = df.subfolder.map(match)
    use = df[df.our_class.isin(CLASS_NAMES)].copy()
    use["class_folder"] = use.our_class; use["class_id"] = use.class_folder.map(CLASS_NAMES.index)
    use["file_name"] = use.path; use["object_id"] = "BDWaste/" + use.subfolder; use["key"] = use.path
    t0 = time.time()
    for p in use.path:
        with Image.open(p) as im:
            im.draft("RGB", (512, 512)); im = ImageOps.exif_transpose(im).convert("RGB")
            w, h = im.size; s = 256 / min(w, h); BDW_CACHE[p] = np.asarray(im.resize((round(w * s), round(h * s)), Image.LANCZOS))
    print(f"BDWaste: {len(use)} images in {use.class_folder.nunique()} shared classes loaded in {time.time() - t0:.0f} s")
    BDW = use.reset_index(drop=True)
    return BDW

def bdwaste_summary():
    rows = []
    for f in glob.glob(os.path.join(OUT_DIR, "all_bdwaste", "*", "all", "predictions.csv")):
        p = pd.read_csv(f); prob = p[[f"p_{c}" for c in CLASS_NAMES]].values; y = p.y_true.values
        shared = sorted(set(y)); restricted = np.array(shared)[prob[:, shared].argmax(1)]
        m = json.load(open(os.path.join(os.path.dirname(f), "metrics.json")))
        rows.append({"model": m["model"], "images": len(p), "shared_classes": len(shared),
                     "acc_23way": float((p.y_pred.values == y).mean()),
                     "acc_restricted_to_shared": float((restricted == y).mean()),
                     "macro_f1_restricted": f1_score(y, restricted, average="macro", labels=shared, zero_division=0)})
    r = pd.DataFrame(rows); r.to_csv(os.path.join(OUT_DIR, "bdwaste_summary.csv"), index=False); print(r.round(4).to_string(index=False))

# ---------- TFLite export: ONNX in this notebook, conversion in a separate Python process (own TF/onnx2tf versions)
EXPORT_DIR = "/kaggle/working/export" if OUT_DIR.startswith("/kaggle") else os.path.join(os.path.dirname(OUT_DIR), "export")
TFLITE_DIR = os.path.join(os.path.dirname(EXPORT_DIR), "tflite")

def prepare_export(jobs):
    """Train/load each model (fold 0), export ONNX (GELU -> tanh approx.), save fold-0 test + calibration images."""
    os.makedirs(EXPORT_DIR, exist_ok=True)
    ev = eval_frame_for(jobs[0]); tr = split_frames(jobs[0])[0]
    crop = T.Compose([T.Resize(256), T.CenterCrop(224)])
    np.save(os.path.join(EXPORT_DIR, "test_uint8.npy"), np.stack([np.asarray(crop(load_image(k))) for k in (ev.class_folder + "/" + ev.file_name)]))
    np.save(os.path.join(EXPORT_DIR, "test_labels.npy"), ev.class_id.values)
    cal = tr.sample(min(200, len(tr)), random_state=0)
    np.save(os.path.join(EXPORT_DIR, "calib_uint8.npy"), np.stack([np.asarray(crop(load_image(k))) for k in (cal.class_folder + "/" + cal.file_name)]))
    meta = []
    for job in jobs:
        model, mean, std = get_trained(job); model = model.float().cpu().eval()
        def swap(mod):
            c = 0
            for cn, ch in mod.named_children():
                if "GELU" in type(ch).__name__: setattr(mod, cn, nn.GELU(approximate="tanh")); c += 1
                else: c += swap(ch)
            return c
        n = swap(model)
        p = os.path.join(EXPORT_DIR, f"{job['model']}.onnx")
        torch.onnx.export(model, torch.zeros(1, 3, 224, 224), p, input_names=["input"], output_names=["logits"], opset_version=17, dynamo=False)
        meta.append({"model": job["model"], "onnx": p, "mean": list(mean), "std": list(std), "gelu_swapped": n,
                     "params": sum(q.numel() for q in model.parameters())})
        print(f"ONNX exported: {job['model']} (GELU layers swapped: {n})")
        del model; torch.cuda.empty_cache()
    json.dump(meta, open(os.path.join(EXPORT_DIR, "export_meta.json"), "w"), indent=2)

EXPORT_WORKER = r'''
import os, sys, json, glob, shutil, subprocess, numpy as np
if not os.environ.get("SKIP_PIP"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tensorflow==2.21.0", "tf_keras", "onnx==1.20.1",
                    "onnx2tf==2.6.9", "onnxsim", "sng4onnx", "onnx_graphsurgeon", "psutil", "ai_edge_litert"], check=True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf, onnx2tf
from sklearn.metrics import f1_score
ED, TD = sys.argv[1], sys.argv[2]; os.makedirs(TD, exist_ok=True)
X = np.load(os.path.join(ED, "test_uint8.npy")).astype(np.float32) / 255.0; Y = np.load(os.path.join(ED, "test_labels.npy"))
C = np.load(os.path.join(ED, "calib_uint8.npy")).astype(np.float32) / 255.0
rows = []
for m in json.load(open(os.path.join(ED, "export_meta.json"))):
    name = m["model"]; mean = np.array(m["mean"], np.float32); std = np.array(m["std"], np.float32)
    Xn, Cn = (X - mean) / std, (C - mean) / std            # NHWC, normalised
    sm = os.path.join(ED, name + "_saved_model")
    try:
        onnx2tf.convert(input_onnx_file_path=m["onnx"], output_folder_path=sm, non_verbose=True, output_signaturedefs=True,
                        copy_onnx_input_output_names_to_tflite=True, tflite_backend="tf_converter")
    except Exception as e:
        rows.append({"model": name, "variant": "all", "status": "onnx2tf failed: " + str(e)[:200]}); continue
    for kind in ("fp32", "dynamic", "int8"):
        r = {"model": name, "variant": kind}
        try:
            conv = tf.lite.TFLiteConverter.from_saved_model(sm)
            if kind != "fp32": conv.optimizations = [tf.lite.Optimize.DEFAULT]
            if kind == "int8":
                conv.representative_dataset = lambda: ([Cn[i:i + 1]] for i in range(len(Cn)))
                conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                conv.inference_input_type = tf.int8; conv.inference_output_type = tf.int8
            p = os.path.join(TD, f"{name}_{kind}.tflite"); open(p, "wb").write(conv.convert())
            try:
                it = tf.lite.Interpreter(model_path=p, num_threads=4); it.allocate_tensors(); xnn = True
            except Exception:
                it = tf.lite.Interpreter(model_path=p, num_threads=4,
                     experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
                it.allocate_tensors(); xnn = False
            i_d, o_d = it.get_input_details()[0], it.get_output_details()[0]; pred = []
            for i in range(len(Xn)):
                x = Xn[i:i + 1]
                if i_d["dtype"] == np.int8:
                    s, z = i_d["quantization"]; x = np.clip(np.round(x / s + z), -128, 127).astype(np.int8)
                it.set_tensor(i_d["index"], x); it.invoke(); pred.append(int(it.get_tensor(o_d["index"]).argmax()))
            pred = np.array(pred)
            r.update({"status": "ok", "size_MB": os.path.getsize(p) / 1e6, "accuracy": float((pred == Y).mean()),
                      "macro_f1": float(f1_score(Y, pred, average="macro")), "xnnpack_ok": xnn, "file": os.path.basename(p)})
        except Exception as e:
            r["status"] = "failed: " + str(e)[:200]
        rows.append(r); print(r, flush=True)
import pandas as pd
pd.DataFrame(rows).to_csv(os.path.join(TD, "tflite_results.csv"), index=False)
'''

def run_export_worker():
    w = os.path.join(os.path.dirname(EXPORT_DIR), "export_worker.py"); open(w, "w").write(EXPORT_WORKER)
    r = subprocess.run([sys.executable, w, EXPORT_DIR, TFLITE_DIR])
    res = os.path.join(TFLITE_DIR, "tflite_results.csv")
    if os.path.exists(res):
        os.makedirs(os.path.join(OUT_DIR, "tflite"), exist_ok=True); shutil.copy(res, os.path.join(OUT_DIR, "tflite", "tflite_results.csv"))
        meta = os.path.join(EXPORT_DIR, "export_meta.json"); shutil.copy(meta, os.path.join(OUT_DIR, "tflite", "export_meta.json"))
        print(pd.read_csv(res).to_string(index=False))
    else:
        print("Export worker did not produce results; return code", r.returncode)

# ===================== CELL: RUN =====================
os.makedirs(OUT_DIR, exist_ok=True)
if "bdwaste" in TASKS:
    build_bdwaste()
for j in JOBS:
    run_job(j)
if "bdwaste" in TASKS:
    bdwaste_summary()
if "shortcut" in TASKS:                                  # degraded-image test (+ Grad-CAM on fold 0)
    for j in SHORTCUT_JOBS:
        m, mean, std = get_trained(j); name, sub = run_dir(j)
        out = os.path.join(OUT_DIR, "shortcut", name, sub)
        shortcut_eval(m, mean, std, eval_frame_for(j), out)
        if j.get("fold") == 0:
            gradcam_report(m, mean, std, eval_frame_for(j), os.path.join(OUT_DIR, "gradcam", name, sub), f"{j['model']} fold 0")
        del m; torch.cuda.empty_cache()
if "export" in TASKS:                                    # phone files (.tflite) + their accuracy on the fold-0 test set
    prepare_export(EXPORT_JOBS)
    run_export_worker()

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
        if os.path.isfile(f) and not f.endswith((".pt", ".onnx", ".tflite", ".npy")) and "_saved_model" not in f:
            zf.write(f, os.path.relpath(f, os.path.dirname(OUT_DIR)))
print("Download this file from the Output tab:", zip_path)
if os.path.isdir(TFLITE_DIR) and glob.glob(os.path.join(TFLITE_DIR, "*.tflite")):
    tz = os.path.join(os.path.dirname(OUT_DIR), f"tflite_{SLOT}.zip")
    with zipfile.ZipFile(tz, "w", zipfile.ZIP_STORED) as zf:
        for f in glob.glob(os.path.join(TFLITE_DIR, "*")): zf.write(f, os.path.basename(f))
    print("Phone files (.tflite) for the Realme benchmark:", tz)
