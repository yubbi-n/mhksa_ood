# KPCA vs Ours (H = 10 for Ours)

FPR95 (%) per OOD set, lower is better; Avg = mean over OOD sets (± std over seeds); AUROC (%) higher is better.

## CE · heads γ = median γ × [0.3, 10]

| | method | SVHN | LSUN | iSUN | Texture | Places365 | Avg FPR95 ↓ | Avg AUROC ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KPCA | median γ (H = 1) | 40.49 | 24.91 | 41.69 | 29.26 | 65.06 | 40.28 ± 2.66 | 92.24 |
|  | tuned γ=2 (H = 1) | 20.80 | 20.87 | 24.19 | 22.19 | 55.08 | 28.63 ± 0.68 | 94.81 |
| Ours | MHKSA (H = 10) | 22.69 | 19.11 | 21.02 | 22.90 | 51.25 | **27.39 ± 0.29** | 95.03 |
|  | Ensemble (z-score) (H = 10) | 19.44 | 19.41 | 27.06 | 21.68 | 57.63 | 29.04 ± 0.19 | 94.57 |

## SupCon · heads γ = median γ × [0.3, 10]

| | method | SVHN | LSUN | iSUN | Texture | Places365 | Avg FPR95 ↓ | Avg AUROC ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KPCA | median γ (H = 1) | 0.75 | 1.12 | 22.05 | 6.10 | 23.19 | 10.64 ± 1.91 | 97.85 |
|  | tuned γ=1 (H = 1) | 0.78 | 0.89 | 15.57 | 4.87 | 18.62 | **8.14 ± 0.24** | 98.30 |
| Ours | MHKSA (H = 10) | 0.75 | 1.18 | 20.65 | 5.38 | 21.83 | 9.96 ± 0.31 | 98.12 |
|  | Ensemble (z-score) (H = 10) | 0.60 | 0.77 | 17.34 | 4.79 | 19.63 | 8.63 ± 0.24 | 98.20 |

