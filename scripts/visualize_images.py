"""Image-level views of the OOD scores.

  examples : grids of real images picked by score (no torch model needed; torchvision only for loading images)
             - OOD images the method flags most / least confidently
             - OOD images the baseline misses but the method catches ("fixed") and the reverse ("broken")
             - ID test images the method wrongly flags as OOD
  heatmap  : Grad-CAM of the OOD score on ResNet18 feature maps (needs torch + the ood-kernel-pca repo/checkpoint)
             image | baseline map | method map, for images chosen as in `examples`

    python scripts/visualize_images.py examples --config configs/c10_r18_ce.yaml \
        --set data.cache_dir=$HOME/ood-kernel-pca/cache --H 2 --data_root ~/data
    python scripts/visualize_images.py heatmap  --config configs/c10_r18_ce.yaml \
        --set data.cache_dir=$HOME/ood-kernel-pca/cache --H 2 --data_root ~/data \
        --kpca_repo ~/ood-kernel-pca --ckpt ~/ood-kernel-pca/save/CIFAR10/R18/ce/checkpoint_100.pth.tar

Feature files are written by ood-kernel-pca/feat_extract.py with shuffle=False, so row i of
<ood>_out.npy is image i of the same torchvision dataset; `heatmap` re-extracts features from the images and
reports how well they match the cache, which also checks that alignment.
Baseline = head0 of the H=1 checkpoint (median-heuristic gamma, CoRP PCA ratio), same seed.
"""
import os
import sys
import csv
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import load_config, run_name          # noqa: E402
from src.data import load_eval                         # noqa: E402
from src.utils import ckpt_path, load_pickle           # noqa: E402

SURFACE, INK, INK2 = '#fcfcfb', '#0b0b0b', '#52514e'
GOOD, BAD = '#0ca30c', '#d03b3b'                       # status colours: caught / missed (always with a text label)
MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)   # ood-kernel-pca CIFAR10 test transform
plt.rcParams.update({'figure.facecolor': SURFACE, 'savefig.facecolor': SURFACE, 'text.color': INK,
                     'font.size': 9})


# ---------------------------------------------------------------- data
IMG_EXT = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')   # torchvision's list


class _Images:
    """torch-free reader with the same files / order as the torchvision datasets ood-kernel-pca uses."""

    def __init__(self, name, root):
        from PIL import Image
        self.Image = Image
        if name == 'CIFAR10':                              # torchvision CIFAR10(train=False): test_batch
            import pickle
            with open(os.path.join(root, 'CIFAR10', 'cifar-10-batches-py', 'test_batch'), 'rb') as f:
                d = pickle.load(f, encoding='latin1')
            self.arr = d['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        elif name == 'SVHN':                               # torchvision SVHN(split='test'): test_32x32.mat
            from scipy.io import loadmat
            self.arr = loadmat(os.path.join(root, 'ood_data', 'svhn', 'test_32x32.mat'))['X'].transpose(3, 0, 1, 2)
        else:                                              # torchvision ImageFolder ordering
            d = os.path.join(root, 'ood_data', 'dtd', 'images') if name == 'Texture' else os.path.join(root, 'ood_data', name)
            self.arr, self.files = None, []
            for c in sorted(e.name for e in os.scandir(d) if e.is_dir()):
                for r, _, fs in sorted(os.walk(os.path.join(d, c), followlinks=True)):
                    self.files += [os.path.join(r, f) for f in sorted(fs) if f.lower().endswith(IMG_EXT)]

    def __len__(self):
        return len(self.arr) if self.arr is not None else len(self.files)

    def __getitem__(self, i):
        im = (self.Image.fromarray(self.arr[i]) if self.arr is not None
              else self.Image.open(self.files[i]).convert('RGB'))
        return im.resize((32, 32), self.Image.BILINEAR), 0


def image_dataset(name, root):
    """same datasets / order as ood-kernel-pca/utils_ood.py; images resized to 32x32 (what the network saw)."""
    root = os.path.expanduser(root)
    try:
        import torchvision as tv
        import torchvision.transforms as T
    except Exception as e:                                 # no / broken torch: read the files directly
        print(f'[{name}] torchvision unavailable ({type(e).__name__}); reading image files directly')
        return _Images(name, root)
    tf = T.Resize((32, 32))
    if name == 'CIFAR10':
        return tv.datasets.CIFAR10(os.path.join(root, 'CIFAR10'), train=False, transform=tf, download=False)
    if name == 'SVHN':
        return tv.datasets.SVHN(os.path.join(root, 'ood_data', 'svhn'), split='test', transform=tf, download=False)
    if name == 'Texture':
        return tv.datasets.ImageFolder(os.path.join(root, 'ood_data', 'dtd', 'images'), transform=tf)
    return tv.datasets.ImageFolder(os.path.join(root, 'ood_data', name), transform=tf)


def get_image(ds, i):
    return np.asarray(ds[int(i)][0].convert('RGB'))


def setup(a):
    cfg = load_config(a.config, a.set)
    name = run_name(cfg)
    model = load_pickle(ckpt_path(cfg, name, a.seed, a.H))['model']
    base = load_pickle(ckpt_path(cfg, name, a.seed, 1))['model']
    f_in, f_out = load_eval(cfg)
    out = os.path.join(a.out, f'{cfg.data.train_mode}_seed{a.seed}_H{a.H}_{a.method}')
    os.makedirs(out, exist_ok=True)
    return cfg, model, base, f_in, f_out, out


def scores_for(model, base, method, x):
    """higher = more ID (same convention as evaluate.py)."""
    return model.scores(x)[method], base.scores(x)['head0']


def pick(f_in, f_out, model, base, method, k):
    """returns {set: {group: [(index, method score pct, baseline score pct, detected_m, detected_b), ...]}}"""
    s_in, b_in = scores_for(model, base, method, f_in)
    thr_m, thr_b = np.percentile(s_in, 5), np.percentile(b_in, 5)      # 95% of ID kept as ID
    pct = lambda s, ref: 100.0 * np.searchsorted(np.sort(ref), s) / len(ref)
    groups, summary = {}, []
    for n, x in f_out.items():
        s, b = scores_for(model, base, method, x)
        dm, db = s < thr_m, b < thr_b
        p_m, p_b = pct(s, s_in), pct(b, b_in)
        row = lambda idx: [(int(i), p_m[i], p_b[i], dm[i], db[i]) for i in idx]
        fixed, broken = np.where(dm & ~db)[0], np.where(~dm & db)[0]
        groups[n] = {
            'most OOD-like': row(np.argsort(s)[:k]),
            'least OOD-like (hardest)': row(np.argsort(-s)[:k]),
            'fixed: baseline missed, method caught': row(fixed[np.argsort((s - thr_m)[fixed])][:k]),
            'broken: baseline caught, method missed': row(broken[np.argsort((thr_m - s)[broken])][:k]),
        }
        summary.append((n, len(x), dm.mean() * 100, db.mean() * 100, len(fixed), len(broken)))
    false_alarm = np.argsort(s_in)[:k]
    groups['CIFAR10'] = {'ID test flagged as OOD (false alarms)':
                         [(int(i), pct(s_in, s_in)[i], pct(b_in, b_in)[i], True, b_in[i] < thr_b) for i in false_alarm]}
    return groups, summary


def write_tables(groups, summary, out, method):
    with open(os.path.join(out, 'picked.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'group', 'index', f'{method}_ID_percentile', 'baseline_ID_percentile',
                    f'{method}_detected', 'baseline_detected'])
        for n, gs in groups.items():
            for g, rows in gs.items():
                for r in rows:
                    w.writerow([n, g, r[0], f'{r[1]:.1f}', f'{r[2]:.1f}', bool(r[3]), bool(r[4])])
    with open(os.path.join(out, 'summary.txt'), 'w') as f:
        f.write(f'threshold = 95% TPR on ID test; detected = flagged as OOD\n')
        f.write(f'{"OOD set":10s} {"n":>6s} {method + " det%":>14s} {"baseline det%":>14s} {"fixed":>7s} {"broken":>7s}\n')
        for n, cnt, dm, db, fx, br in summary:
            f.write(f'{n:10s} {cnt:6d} {dm:14.2f} {db:14.2f} {fx:7d} {br:7d}\n')
    print(open(os.path.join(out, 'summary.txt')).read())


def caption(ax, r, method):
    _, pm, pb, dm, db = r
    ax.set_title(f'{method[:5]} {pm:4.1f}%  {"OOD" if dm else "ID"}\nbase  {pb:4.1f}%  {"OOD" if db else "ID"}',
                 fontsize=7, color=INK2, loc='left', family='monospace')


# ---------------------------------------------------------------- examples
def cmd_examples(a):
    cfg, model, base, f_in, f_out, out = setup(a)
    groups, summary = pick(f_in, f_out, model, base, a.method, a.k)
    write_tables(groups, summary, out, a.method)
    for n, gs in groups.items():
        ds = image_dataset(n, a.data_root)
        rows = [(g, r) for g, r in gs.items() if r]
        if not rows:
            continue
        fig, axes = plt.subplots(len(rows), a.k, figsize=(1.35 * a.k, 1.75 * len(rows) + 0.6), squeeze=False)
        for i, (g, rs) in enumerate(rows):
            for j in range(a.k):
                ax = axes[i, j]
                ax.axis('off')
                if j < len(rs):
                    ax.imshow(get_image(ds, rs[j][0]), interpolation='nearest')
                    caption(ax, rs[j], a.method)
            axes[i, 0].text(-0.08, 0.5, g, transform=axes[i, 0].transAxes, rotation=90, ha='right', va='center',
                            fontsize=8, color=INK)
        fig.suptitle(f'{cfg.data.train_mode.upper()} · {n} · {a.method} (H={a.H}) vs baseline (H=1), seed {a.seed}\n'
                     'number = percentile of the score among ID test scores (low = OOD-like); '
                     'OOD/ID = decision at 95% TPR', fontsize=9, color=INK)
        fig.tight_layout(rect=(0.03, 0, 1, 0.93))
        path = os.path.join(out, f'examples_{n}.png')
        fig.savefig(path, dpi=180)
        plt.close(fig)
        print('saved', path)


# ---------------------------------------------------------------- heatmap
class TorchScorer:
    """OOD-ness (higher = more OOD) of a pooled feature, differentiable; mirrors src/model.py."""

    def __init__(self, model, method, torch, device):
        t = lambda v: torch.as_tensor(np.asarray(v), dtype=torch.float32, device=device)
        hs = model.stage1.heads
        self.torch, self.method = torch, method
        self.W, self.u = [t(h.rff.w) for h in hs], [t(h.rff.u) for h in hs]
        self.M = [h.rff.M for h in hs]
        self.mu = [t(h.pca.mu) for h in hs]
        self.U, self.Umh = [t(h.pca.U) for h in hs], [t(h.pca_mh.U) for h in hs]
        p2 = model.mhksa.pca2
        self.mu2, self.U2 = t(p2.mu), t(p2.U)

    def __call__(self, z):
        torch = self.torch
        z = z / (z.norm(dim=1, keepdim=True) + 1e-10)
        res2 = lambda xc, U: (xc ** 2).sum(1) - ((xc @ U) ** 2).sum(1)
        xcs = [np.sqrt(2.0 / M) * torch.cos(z @ W.T + u) - mu
               for W, u, M, mu in zip(self.W, self.u, self.M, self.mu)]
        if self.method.startswith('head'):
            k = int(self.method[4:])
            return res2(xcs[k], self.U[k]).clamp_min(0).sqrt()
        e1 = sum(res2(xc, U) for xc, U in zip(xcs, self.Umh)).clamp_min(0)
        c = torch.cat([xc @ U for xc, U in zip(xcs, self.Umh)], 1) - self.mu2
        e2 = res2(c, self.U2).clamp_min(0)
        return {'MHKSA': (e1 + e2).sqrt(), 'MHKSA_e1only': e1.sqrt(), 'MHKSA_e2only': e2.sqrt()}[self.method]


def load_net(a, cfg, torch, device):
    sys.path.insert(0, os.path.expanduser(a.kpca_repo))
    from model import resnet, resnet_supcon
    supcon = cfg.data.train_mode == 'supcon'
    net = (resnet_supcon if supcon else resnet).resnet18_cifar()
    try:
        ck = torch.load(os.path.expanduser(a.ckpt), map_location='cpu', weights_only=False)
    except TypeError:                                   # torch < 1.13
        ck = torch.load(os.path.expanduser(a.ckpt), map_location='cpu')
    net.load_state_dict(ck['state_dict'] if 'state_dict' in ck else ck)
    return net.to(device).eval()


def forward_maps(net, x, layer):
    out = net.layer1(__import__('torch').nn.functional.relu(net.bn1(net.conv1(x))))
    out = net.layer2(out)
    a3 = net.layer3(out)
    a4 = net.layer4(a3)
    z = net.avgpool(a4).flatten(1)
    return (a3 if layer == 3 else a4), z


def gradcam(net, scorer, x, layer, torch):
    A, z = forward_maps(net, x, layer)
    s = scorer(z)
    g, = torch.autograd.grad(s.sum(), A)
    cam = torch.relu((g.mean((2, 3), keepdim=True) * A).sum(1, keepdim=True))       # regions raising OOD-ness
    cam = torch.nn.functional.interpolate(cam, size=x.shape[-2:], mode='bilinear', align_corners=False)[:, 0]
    cam = cam / (cam.flatten(1).amax(1)[:, None, None] + 1e-12)
    return cam.detach().cpu().numpy(), s.detach().cpu().numpy(), z.detach().cpu().numpy()


def cmd_heatmap(a):
    import torch
    import torchvision.transforms as T
    cfg, model, base, f_in, f_out, out = setup(a)
    device = 'cuda' if torch.cuda.is_available() and not a.cpu else 'cpu'
    net = load_net(a, cfg, torch, device)
    sc_m, sc_b = TorchScorer(model, a.method, torch, device), TorchScorer(base, 'head0', torch, device)
    to_t = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    groups, summary = pick(f_in, f_out, model, base, a.method, a.k)
    write_tables(groups, summary, out, a.method)

    # sanity check: features re-extracted from images == cached features (same preprocessing and index order)
    ds = image_dataset('CIFAR10', a.data_root)
    idx = list(range(min(64, len(ds))))
    x = torch.stack([to_t(ds[i][0]) for i in idx]).to(device)
    with torch.no_grad():
        _, z = forward_maps(net, x, 4)
    zc = z.cpu().numpy()
    zc /= np.linalg.norm(zc, axis=1, keepdims=True) + 1e-10
    err = np.abs(zc - f_in[idx]).max()
    s_np = -model.scores(f_in[idx])[a.method]
    s_t = sc_m(torch.as_tensor(f_in[idx], device=device)).detach().cpu().numpy()
    msg = (f'check: max |feature(image) - cached feature| = {err:.2e} (normalised); '
           f'max |torch score - numpy score| = {np.abs(s_np - s_t).max():.2e}')
    print(msg)
    if err > 1e-3:
        print('WARNING: features re-extracted from the images do not match the cache -> wrong checkpoint, '
              'data path, or image order; heatmaps would not correspond to the scored images.')

    for n, gs in groups.items():
        ds = image_dataset(n, a.data_root)
        cands = [k for g in a.groups for k in gs if k.startswith(g) and gs[k]] + [k for k in gs if gs[k]]
        key = cands[0] if cands else None
        rs = gs[key][:a.k] if key else []
        if not rs:
            continue
        x = torch.stack([to_t(ds[r[0]][0]) for r in rs]).to(device)
        cam_m, _, _ = gradcam(net, sc_m, x, a.layer, torch)
        cam_b, _, _ = gradcam(net, sc_b, x, a.layer, torch)
        fig, axes = plt.subplots(len(rs), 3, figsize=(5.0, 1.55 * len(rs) + 0.9), squeeze=False)
        for i, r in enumerate(rs):
            img = get_image(ds, r[0])
            for j, (title, cam) in enumerate([('image', None), ('baseline', cam_b[i]), (a.method, cam_m[i])]):
                ax = axes[i, j]
                ax.axis('off')
                ax.imshow(img, interpolation='nearest')
                if cam is not None:
                    ax.imshow(cam, cmap='inferno', alpha=0.55, vmin=0, vmax=1, interpolation='bilinear')
                if i == 0:
                    ax.set_title(title, fontsize=8, color=INK)
            dm, db = r[3], r[4]
            axes[i, 0].text(-0.08, 0.5, f'#{r[0]}\nbase {"OOD" if db else "ID"}\n{a.method[:5]} {"OOD" if dm else "ID"}',
                            transform=axes[i, 0].transAxes, ha='right', va='center', fontsize=7, color=INK2)
        fig.suptitle(f'{cfg.data.train_mode.upper()} · {n} · {key}\n'
                     f'Grad-CAM of the OOD score, layer{a.layer}\nbright = raises OOD score (per-map scale)',
                     fontsize=8, color=INK)
        fig.tight_layout(rect=(0.14, 0, 1, 1 - 0.55 / (1.55 * len(rs) + 0.9)))
        path = os.path.join(out, f'heatmap_{n}.png')
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print('saved', path)
    open(os.path.join(out, 'heatmap_check.txt'), 'w').write(msg + '\n')


def cmd_compare(a):
    """one image, Grad-CAM for every run: baseline | method at H = 1 .. 10 (same seed)."""
    import torch
    import torchvision.transforms as T
    cfg = load_config(a.config, a.set)
    name = run_name(cfg)
    device = 'cuda' if torch.cuda.is_available() and not a.cpu else 'cpu'
    net = load_net(a, cfg, torch, device)
    f_in, f_out = load_eval(cfg)
    feats = dict(f_out, CIFAR10=f_in)
    heads = [h for h in (a.heads or cfg.kernel.heads) if os.path.exists(ckpt_path(cfg, name, a.seed, h))]
    models = {h: load_pickle(ckpt_path(cfg, name, a.seed, h))['model'] for h in heads}
    base = models[1] if 1 in models else load_pickle(ckpt_path(cfg, name, a.seed, 1))['model']

    ds_name, idx = a.dataset, a.index
    if idx is None:                                      # auto: an OOD image baseline misses but method@pick_H catches
        groups, _ = pick(f_in, {ds_name: feats[ds_name]} if ds_name != 'CIFAR10' else {}, models[a.pick_H], base,
                         a.method, 1)
        g = groups[ds_name] if ds_name in groups else {}
        for key in ('fixed: baseline missed, method caught', 'least OOD-like (hardest)', 'most OOD-like'):
            if g.get(key):
                idx, why = g[key][0][0], key
                break
        else:
            idx, why = 0, 'index 0'
        print(f'auto-picked {ds_name} #{idx} ({why} at H={a.pick_H})')

    ds = image_dataset(ds_name, a.data_root)
    to_t = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    x = to_t(ds[idx][0])[None].to(device)
    img = get_image(ds, idx)
    with torch.no_grad():
        _, z = forward_maps(net, x, 4)
    zc = z.cpu().numpy()[0]
    zc /= np.linalg.norm(zc) + 1e-10
    err = np.abs(zc - feats[ds_name][idx]).max()
    print(f'check: |feature(image) - cached feature| = {err:.2e}' + ('  WARNING: mismatch' if err > 1e-3 else ''))

    s_in_cache = {}
    def verdict(model, method):                           # ID-percentile of this image + decision at 95% TPR
        key = (id(model), method)
        if key not in s_in_cache:
            s_in_cache[key] = np.sort(model.scores(f_in)[method])
        ref = s_in_cache[key]
        s = model.scores(feats[ds_name][idx:idx + 1])[method][0]
        return 100.0 * np.searchsorted(ref, s) / len(ref), s < np.percentile(ref, 5)

    cols = [('baseline\n(H=1, median γ)', base, 'head0')] + [(f'{a.method}\nH={h}', models[h], a.method) for h in heads]
    fig, axes = plt.subplots(1, len(cols) + 1, figsize=(max(1.45 * (len(cols) + 1), 7.5), 2.8), squeeze=False)
    ax = axes[0, 0]
    ax.imshow(img, interpolation='nearest')
    ax.set_title(f'{ds_name} #{idx}', fontsize=8, color=INK)
    ax.axis('off')
    for j, (title, m, meth) in enumerate(cols, 1):
        cam, _, _ = gradcam(net, TorchScorer(m, meth, torch, device), x, a.layer, torch)
        p, det = verdict(m, meth)
        ax = axes[0, j]
        ax.imshow(img, interpolation='nearest')
        ax.imshow(cam[0], cmap='inferno', alpha=0.55, vmin=0, vmax=1, interpolation='bilinear')
        ax.set_title(title, fontsize=7.5, color=INK)
        ax.text(0.5, -0.08, f'{"OOD" if det else "ID"} · {p:.1f}%', transform=ax.transAxes, ha='center', va='top',
                fontsize=7.5, color=GOOD if det == (ds_name != 'CIFAR10') else BAD, fontweight='bold')
        ax.axis('off')
    fig.suptitle(f'{cfg.data.train_mode.upper()} · seed {a.seed} · Grad-CAM of the OOD score (layer{a.layer}); '
                 f'bright = raises OOD score\nbelow each map: decision at 95% TPR · ID-percentile of the score '
                 f'(green = correct, red = wrong)', fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    out = os.path.join(a.out, f'{cfg.data.train_mode}_seed{a.seed}_{a.method}')
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f'compare_{ds_name}_{idx}.png')
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print('saved', path)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = 'compare'
    p = sub.add_parser(c, help='one image, heatmap of every run (baseline, H=1..10)')
    p.add_argument('--config', required=True)
    p.add_argument('--set', nargs='*', default=[])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--method', default='MHKSA')
    p.add_argument('--dataset', default='SVHN', help='SVHN | LSUN | iSUN | Texture | places365 | CIFAR10')
    p.add_argument('--index', type=int, default=None, help='image index in that dataset (default: auto-pick)')
    p.add_argument('--pick_H', type=int, default=2, help='H used for the auto-pick')
    p.add_argument('--heads', type=int, nargs='*', default=None, help='H values to show (default: all)')
    p.add_argument('--data_root', default='~/data')
    p.add_argument('--kpca_repo', default='~/ood-kernel-pca')
    p.add_argument('--ckpt', required=True)
    p.add_argument('--layer', type=int, default=3, choices=[3, 4])
    p.add_argument('--out', default='figures/images')
    p.add_argument('--cpu', action='store_true')
    for c in ('examples', 'heatmap'):
        p = sub.add_parser(c)
        p.add_argument('--config', required=True)
        p.add_argument('--set', nargs='*', default=[])
        p.add_argument('--seed', type=int, default=0)
        p.add_argument('--H', type=int, default=2)
        p.add_argument('--method', default='MHKSA', help='MHKSA | MHKSA_e1only | MHKSA_e2only | head<k>')
        p.add_argument('--data_root', default='~/data', help='root used by ood-kernel-pca (CIFAR10/, ood_data/)')
        p.add_argument('--k', type=int, default=8 if c == 'examples' else 4, help='images per row / per set')
        p.add_argument('--out', default='figures/images')
        if c == 'heatmap':
            p.add_argument('--kpca_repo', default='~/ood-kernel-pca')
            p.add_argument('--ckpt', required=True, help='ResNet18 checkpoint used for feat_extract.py')
            p.add_argument('--layer', type=int, default=3, choices=[3, 4], help='3: 8x8 maps, 4: 4x4 maps')
            p.add_argument('--groups', nargs='*', default=['fixed', 'least OOD-like', 'ID test flagged as OOD'],
                           help='which picked group to explain per set (first one that exists)')
            p.add_argument('--cpu', action='store_true')
    a = ap.parse_args()
    {'examples': cmd_examples, 'heatmap': cmd_heatmap, 'compare': cmd_compare}[a.cmd](a)


if __name__ == '__main__':
    main()
