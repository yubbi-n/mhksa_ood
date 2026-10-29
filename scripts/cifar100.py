"""CIFAR-100 / ResNet18 (CE): train the backbone, then extract the features read by src/data.py.

The ood-kernel-pca repo only covers CIFAR10 and ImageNet, so this mirrors its CIFAR10 pipeline for CIFAR-100:
same network (ood-kernel-pca/model/resnet.py resnet18_cifar), same OOD sets and folders (~/data/ood_data, as in
its utils_ood.py), same feature (layer4 output, average-pooled, 512-d) and the same cache layout.

    python scripts/cifar100.py train   --kpca_repo ~/ood-kernel-pca --data_root ~/data --gpu 0
    python scripts/cifar100.py extract --kpca_repo ~/ood-kernel-pca --data_root ~/data --cache ~/ood-kernel-pca/cache --gpu 0

train   : SGD (lr 0.1, momentum 0.9, nesterov, weight decay 5e-4), cosine schedule, batch 128, 100 epochs,
          random crop (pad 4) + horizontal flip -> <kpca_repo>/save/CIFAR100/R18/ce/checkpoint_100.pth.tar
          ({'state_dict': ...}, as the CIFAR10 checkpoints)
extract : <cache>/CIFAR100/R18/ce/{train_in, val_in, SVHN_out, LSUN_out, iSUN_out, Texture_out, places365_out}.npy,
          shape (512, n) float32 like ood-kernel-pca/feat_extract.py; existing files are skipped.
Normalisation uses the CIFAR-100 mean / std for training and extraction alike.
"""
import os
import sys
import argparse

import numpy as np

MEAN, STD = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)        # CIFAR-100 train statistics
OOD = ['SVHN', 'LSUN', 'iSUN', 'Texture', 'places365']


def build_net(kpca_repo):
    sys.path.insert(0, os.path.join(os.path.expanduser(kpca_repo), 'model'))
    import resnet                                                     # ood-kernel-pca/model/resnet.py
    return resnet.resnet18_cifar(num_classes=100)


def transforms(train):
    import torchvision.transforms as T
    aug = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()] if train else [T.Resize((32, 32))]
    return T.Compose(aug + [T.ToTensor(), T.Normalize(MEAN, STD)])


def ckpt_file(kpca_repo, epochs):
    return os.path.join(os.path.expanduser(kpca_repo), 'save', 'CIFAR100', 'R18', 'ce', f'checkpoint_{epochs}.pth.tar')


def train(a):
    import torch
    import torch.nn.functional as F
    import torchvision as tv
    dev = torch.device(f'cuda:{a.gpu}')
    root = os.path.join(os.path.expanduser(a.data_root), 'CIFAR100')
    tr = tv.datasets.CIFAR100(root, train=True, transform=transforms(True), download=True)
    te = tv.datasets.CIFAR100(root, train=False, transform=transforms(False), download=True)
    trl = torch.utils.data.DataLoader(tr, batch_size=128, shuffle=True, num_workers=a.workers, drop_last=True)
    tel = torch.utils.data.DataLoader(te, batch_size=512, num_workers=a.workers)
    torch.manual_seed(a.seed)
    net = build_net(a.kpca_repo).to(dev)
    opt = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * len(trl))
    for ep in range(1, a.epochs + 1):
        net.train()
        for x, y in trl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            loss = F.cross_entropy(net(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        if ep % 10 == 0 or ep == a.epochs:
            net.eval()
            with torch.no_grad():
                acc = sum((net(x.to(dev)).argmax(1).cpu() == y).sum().item() for x, y in tel) / len(te)
            print(f'epoch {ep}/{a.epochs}  loss {loss.item():.3f}  test acc {acc * 100:.2f}%', flush=True)
    path = ckpt_file(a.kpca_repo, a.epochs)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({'state_dict': net.state_dict(), 'epoch': a.epochs, 'test_acc': acc}, path)
    print('saved', path)


def features(net, loader, dev):
    import torch
    import torch.nn.functional as F
    out = []
    with torch.no_grad():
        for x, _ in loader:
            _, fl = net.feature_list(x.to(dev))
            out.append(torch.flatten(F.adaptive_avg_pool2d(fl[-1], (1, 1)), 1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def extract(a):
    import torch
    import torchvision as tv
    dev = torch.device(f'cuda:{a.gpu}')
    net = build_net(a.kpca_repo)
    net.load_state_dict(torch.load(a.ckpt or ckpt_file(a.kpca_repo, a.epochs), map_location='cpu')['state_dict'])
    net = net.to(dev).eval()
    data = os.path.expanduser(a.data_root)
    tf = transforms(False)
    sets = {'train_in': lambda: tv.datasets.CIFAR100(os.path.join(data, 'CIFAR100'), train=True, transform=tf, download=True),
            'val_in': lambda: tv.datasets.CIFAR100(os.path.join(data, 'CIFAR100'), train=False, transform=tf, download=True),
            'SVHN_out': lambda: tv.datasets.SVHN(os.path.join(data, 'ood_data', 'svhn'), split='test', transform=tf, download=False),
            'Texture_out': lambda: tv.datasets.ImageFolder(os.path.join(data, 'ood_data', 'dtd', 'images'), transform=tf)}
    for n in ['LSUN', 'iSUN', 'places365']:
        sets[f'{n}_out'] = (lambda n=n: tv.datasets.ImageFolder(os.path.join(data, 'ood_data', n), transform=tf))
    out_dir = os.path.join(os.path.expanduser(a.cache), 'CIFAR100', 'R18', 'ce')
    os.makedirs(out_dir, exist_ok=True)
    for name, make in sets.items():
        path = os.path.join(out_dir, f'{name}.npy')
        if os.path.exists(path):
            print('exists, skipped', path)
            continue
        ds = make()
        f = features(net, torch.utils.data.DataLoader(ds, batch_size=512, num_workers=a.workers), dev)
        np.save(path, np.ascontiguousarray(f.T))                     # (512, n), as feat_extract.py
        print(f'saved {path} {f.T.shape}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('what', choices=['train', 'extract'])
    ap.add_argument('--kpca_repo', default='~/ood-kernel-pca')
    ap.add_argument('--data_root', default='~/data', help='holds CIFAR100/ (downloaded if missing) and ood_data/')
    ap.add_argument('--cache', default='~/ood-kernel-pca/cache', help='data.cache_dir of the config')
    ap.add_argument('--ckpt', default=None, help='[extract] checkpoint (default: the one train writes)')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    {'train': train, 'extract': extract}[a.what](a)


if __name__ == '__main__':
    main()
