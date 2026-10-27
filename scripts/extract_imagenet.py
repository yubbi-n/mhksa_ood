"""ImageNet-1K / ResNet50 (torchvision, CE) penultimate features for the cache read by src/data.py.

Server disk is small, so ImageNet is streamed from the Hugging Face hub one parquet shard at a time
(download -> features -> delete). OOD sets are plain image folders (any layout, searched recursively).

    # 0) once: accept the terms on https://huggingface.co/datasets/ILSVRC/imagenet-1k, then  huggingface-cli login
    python scripts/extract_imagenet.py train --cache ./cache --per-class 200 --gpu 0
    python scripts/extract_imagenet.py val   --cache ./cache --gpu 0
    python scripts/extract_imagenet.py ood   --cache ./cache --gpu 0 --ood iNaturalist=/path/iNaturalist SUN=/path/SUN \
                                             Places=/path/Places Textures=/path/dtd/images

Writes <cache>/ImageNet/R50/ce/{train_in, val_in, <name>_out}.npy, shape (2048, n), float16 (src/data.py casts
to float32 and transposes), plus {train,val}_labels.npy. Per-shard parts go to <cache>/ImageNet/R50/ce/parts/ and
make a rerun resume where it stopped; delete that folder when done.
Preprocessing: resize 256, center crop 224, ImageNet mean/std (torchvision IMAGENET1K_V1 weights).
--per-class N keeps the first N training images of every class (in shard order) and stops once all classes are full.
"""
import os
import io
import glob
import shutil
import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

REPO = 'ILSVRC/imagenet-1k'
IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.JPEG', '.JPG', '.PNG')


def build_model(gpu):
    import torch
    import torchvision
    dev = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    m.fc = torch.nn.Identity()                      # penultimate (avgpool) features, 2048-d
    tf = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256), torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return m.eval().to(dev), tf, dev


class Extractor:
    def __init__(self, gpu, batch, workers):
        import torch
        self.torch = torch
        self.model, self.tf, self.dev = build_model(gpu)
        self.batch = batch
        self.pool = ThreadPoolExecutor(workers)

    def _prep(self, item):
        img = Image.open(io.BytesIO(item) if isinstance(item, (bytes, bytearray)) else item).convert('RGB')
        return self.tf(img)

    def __call__(self, items):
        """items: list of image bytes or file paths -> (n, 2048) float16"""
        out = []
        with self.torch.no_grad():
            for i in range(0, len(items), self.batch):
                x = self.torch.stack(list(self.pool.map(self._prep, items[i:i + self.batch]))).to(self.dev)
                out.append(self.model(x).float().cpu().numpy().astype(np.float16))
        return np.concatenate(out) if out else np.zeros((0, 2048), np.float16)


def shard_files(split):
    """parquet shards of `split` on the hub; tries the main branch, then the auto-converted parquet branch."""
    from huggingface_hub import HfApi
    key = {'train': 'train', 'val': 'validation'}[split]
    for rev in ('main', 'refs/convert/parquet'):
        try:
            files = HfApi().list_repo_files(REPO, repo_type='dataset', revision=rev)
        except Exception as e:                      # not logged in / terms not accepted / branch missing
            print(f'[{rev}] cannot list files: {e}')
            continue
        files = sorted(f for f in files if f.endswith('.parquet') and key in os.path.basename(f) + os.path.dirname(f))
        if files:
            print(f'{len(files)} {key} shards on {REPO}@{rev}')
            return rev, files
    raise SystemExit(f'no {key} parquet shards found on {REPO}; check the hub page, the login and the accepted terms')


def run_hub(split, out_dir, ext, per_class, tmp):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    rev, files = shard_files(split)
    parts = os.path.join(out_dir, 'parts')
    os.makedirs(parts, exist_ok=True)
    count = np.zeros(1000, int)
    for i, f in enumerate(files):
        done = os.path.join(parts, f'{split}_{i:04d}.npz')
        if os.path.exists(done):                    # resume
            count += np.bincount(np.load(done)['y'], minlength=1000)
            continue
        if per_class and (count >= per_class).all():
            break
        local = hf_hub_download(REPO, f, repo_type='dataset', revision=rev, local_dir=tmp)
        imgs, ys = [], []
        for b in pq.ParquetFile(local).iter_batches(columns=['image', 'label'], batch_size=1024):
            for img, y in zip(b.column('image').to_pylist(), b.column('label').to_pylist()):
                if per_class and count[y] >= per_class:
                    continue
                count[y] += 1
                imgs.append(img['bytes'])
                ys.append(y)
        feats = ext(imgs)
        np.savez(done, f=feats, y=np.array(ys, np.int16))
        os.remove(local)
        print(f'[{split}] shard {i + 1}/{len(files)}: kept {len(ys)}, total {count.sum()}, '
              f'classes full {(count >= per_class).sum() if per_class else "-"}', flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
    got = sorted(glob.glob(os.path.join(parts, f'{split}_*.npz')))
    F = np.concatenate([np.load(p)['f'] for p in got])
    Y = np.concatenate([np.load(p)['y'] for p in got])
    name = {'train': 'train_in', 'val': 'val_in'}[split]
    np.save(os.path.join(out_dir, f'{name}.npy'), np.ascontiguousarray(F.T))     # (2048, n), as src/data.py expects
    np.save(os.path.join(out_dir, f'{split}_labels.npy'), Y)
    print(f'saved {name}.npy {F.shape[::-1]} ({F.nbytes / 1e9:.2f} GB) and {split}_labels.npy')


def run_ood(specs, out_dir, ext):
    for spec in specs:
        name, root = spec.split('=', 1)
        paths = sorted(p for p in glob.glob(os.path.join(root, '**', '*'), recursive=True) if p.endswith(IMG_EXT))
        if not paths:
            raise SystemExit(f'{name}: no images under {root}')
        F = ext(paths)
        np.save(os.path.join(out_dir, f'{name}_out.npy'), np.ascontiguousarray(F.T))
        print(f'saved {name}_out.npy {F.shape[::-1]}  ({len(paths)} images from {root})')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('what', choices=['train', 'val', 'ood'])
    ap.add_argument('--cache', default='./cache', help='data.cache_dir of the config')
    ap.add_argument('--per-class', type=int, default=200, help='[train] images per class (0 = all 1.28M)')
    ap.add_argument('--ood', nargs='*', default=[], help='[ood] name=folder, e.g. SUN=/data/SUN')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--workers', type=int, default=8, help='decoding threads')
    ap.add_argument('--tmp', default=None, help='where one shard is downloaded (default <cache>/_hf_tmp)')
    a = ap.parse_args()
    out_dir = os.path.join(a.cache, 'ImageNet', 'R50', 'ce')
    os.makedirs(out_dir, exist_ok=True)
    ext = Extractor(a.gpu, a.batch, a.workers)
    if a.what == 'ood':
        run_ood(a.ood, out_dir, ext)
    else:
        run_hub(a.what, out_dir, ext, a.per_class if a.what == 'train' else 0, a.tmp or os.path.join(a.cache, '_hf_tmp'))


if __name__ == '__main__':
    main()
