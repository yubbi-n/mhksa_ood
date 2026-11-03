# CIFAR-100 (ResNet18, CE) — head 범위 ×0.3~10 결과

## 설정
- 백본: ResNet18 (ood-kernel-pca의 resnet18_cifar), CIFAR-100으로 직접 학습 (100 epoch, test acc 77.87%)
- ID train 50,000장 / ID test 10,000장
- OOD: SVHN (26,032), LSUN (10,000), iSUN (8,925), Texture (5,640), Places365 (10,000) — CIFAR-10 실험과 같은 OOD 세트, OOD별 FPR95를 구한 뒤 평균
- CoRP 설정: CIFAR-10과 동일 (M 2048, PCA 비율 0.8). 원논문에 CIFAR-100 결과는 없음
- MHKSA: r1 = r2 = 0.9 (r1 × r2 ≈ 0.8)
- head γ = median γ × geomspace(0.3, 10, H), H = 1~10, seed 5개 (mean ± std)
- γ와 비율 모두 test OOD로 고르지 않은 고정 설정 (CIFAR-10·ImageNet과 같은 규칙)

## KPCA vs Ours — OOD별 (FPR95 ↓, H = 10)
| group | method | SVHN | LSUN | iSUN | Texture | Places365 | Avg FPR95 ↓ | Avg AUROC ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KPCA | median γ (H = 1) | 64.53 | 82.41 | 77.92 | 53.19 | 88.60 | 73.33 ± 1.81 | 78.40 |
| Ours | MHKSA (H = 10) | 57.40 | 71.80 | 67.85 | 55.18 | 81.51 | 66.75 ± 0.47 | 84.19 |
| Ours | Ensemble z-score (H = 10) | 57.63 | 79.73 | 67.83 | 50.90 | 85.51 | 68.32 ± 0.54 | 82.02 |
| ref | best head (oracle, test-picked, H = 10) | 56.59 | 73.23 | 69.31 | 57.77 | 82.78 | 67.94 ± 0.67 | 83.65 |

## H에 따른 변화 (FPR95, AVG)
| H | MHKSA | MHKSA e1만 | MHKSA e2만 | best head | Ens percentile | Ens z-score |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 72.23 | 73.25 | 93.60 | 73.33 | 73.33 | 73.33 |
| 2 | 69.05 | 70.73 | 75.15 | 70.34 | 78.83 | 73.62 |
| 3 | 67.45 | 68.29 | 71.48 | 69.08 | 75.31 | 70.36 |
| 4 | 67.52 | 67.49 | 70.30 | 69.25 | 73.44 | 69.18 |
| 5 | 66.82 | 66.86 | 68.87 | 68.02 | 73.49 | 69.00 |
| 6 | 66.73 | 67.58 | 67.17 | 68.09 | 72.57 | 69.12 |
| 7 | 67.24 | 67.29 | 67.70 | 69.34 | 72.64 | 69.32 |
| 8 | 66.09 | 66.72 | 66.94 | 67.11 | 71.85 | 68.24 |
| 9 | 65.82 | 66.87 | 65.56 | 65.44 | 72.55 | 68.92 |
| 10 | 66.75 | 67.50 | 66.62 | 67.94 | 71.59 | 68.32 |

## 정리
1. **Ours가 median KPCA보다 좋다**: 73.33 → MHKSA 66.75 (−6.6), Ensemble 68.32 (−5.0). AUROC 78.40 → 84.19.
2. **MHKSA가 test OOD로 고른 best head보다도 좋다**: 66.75 vs 67.94 (−1.2). 세 데이터셋 중 head를 섞는 이득이 처음으로 나타남 (CIFAR-10·ImageNet은 best head와 비슷).
3. **H ≥ 5에서 65.8~67.2로 안정** → H 선택에 민감하지 않음.
4. **OOD별**: MHKSA가 5개 중 4개에서 median KPCA보다 좋음. Texture만 약간 나쁨 (55.18 vs 53.19), Texture는 Ensemble이 가장 좋음 (50.90).
5. 원논문에 CIFAR-100 결과가 없어 "튜닝된 KPCA" 기준이 없음 → best head(test OOD로 고른 값)를 그 역할로 사용.
