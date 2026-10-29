# influence: supcon   run=CIFAR10-R18-supcon-evr0.7-mh0.84_0.84-M2048-m0.3_10log   H=10   seeds=[0, 1, 2, 3, 4]

## single heads (FPR95, AVG over OOD sets) and rank (0 = best)

| head | gamma | FPR95 | rank |
| --- | --- | --- | --- |
| 0 | 0.107 | 14.22 | 7 |
| 1 | 0.158 | 15.38 | 9 |
| 2 | 0.234 | 14.81 | 8 |
| 3 | 0.345 | 10.35 | 6 |
| 4 | 0.510 | 9.30 | 1 |
| 5 | 0.753 | 9.56 | 2 |
| 6 | 1.112 | 8.13 | 0 |
| 7 | 1.641 | 9.63 | 3 |
| 8 | 2.423 | 9.80 | 4 |
| 9 | 3.577 | 9.82 | 5 |

sanity: refitted stage 2 on all heads vs checkpoint MHKSA, max |score diff| = 7.75e-07

## rule: MHKSA (stage 2 refitted per head subset)   (all 10 heads: FPR95 9.96)

### ① influence vs contribution, and ② leave-one-head-out

(shares: stage-1 terms e_h^2 only; FPR columns: full MHKSA with the refitted stage 2)

| head | gamma | single FPR | weight on ID | share of OOD-ID gap (signal) | share of ID variance (noise) | FPR change if removed (- = harmful head) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.107 | 14.22 | 2.2% | 2.8% | 0.3% | +0.19 |
| 1 | 0.158 | 15.38 | 3.1% | 3.9% | 0.6% | +0.17 |
| 2 | 0.234 | 14.81 | 4.3% | 5.3% | 1.1% | +0.21 |
| 3 | 0.345 | 10.35 | 6.1% | 7.0% | 2.2% | +0.04 |
| 4 | 0.510 | 9.30 | 8.3% | 9.2% | 4.1% | -0.13 |
| 5 | 0.753 | 9.56 | 10.6% | 11.8% | 7.2% | -0.02 |
| 6 | 1.112 | 8.13 | 13.2% | 14.2% | 11.8% | +0.08 |
| 7 | 1.641 | 9.63 | 15.7% | 16.0% | 18.8% | +0.06 |
| 8 | 2.423 | 9.80 | 17.7% | 16.0% | 25.9% | -0.49 |
| 9 | 3.577 | 9.82 | 18.9% | 13.8% | 28.1% | -2.10 |

### ③ groups (halves by single-head FPR, per seed)

| all 10 heads | better half only | worse half only | damage from the worse half (all - better half) |
| --- | --- | --- | --- |
| 9.96 | 9.83 | 9.51 | +0.13 |

## rule: z-score mean (ensemble)   (all 10 heads: FPR95 8.63)

### ① influence vs contribution, and ② leave-one-head-out

| head | gamma | single FPR | weight on ID | share of OOD-ID gap (signal) | share of ID variance (noise) | FPR change if removed (- = harmful head) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.107 | 14.22 | 10.0% | 9.9% | 8.0% | -0.14 |
| 1 | 0.158 | 15.38 | 10.0% | 9.5% | 7.9% | -0.13 |
| 2 | 0.234 | 14.81 | 10.0% | 9.5% | 8.0% | -0.15 |
| 3 | 0.345 | 10.35 | 10.0% | 11.3% | 9.6% | -0.08 |
| 4 | 0.510 | 9.30 | 10.0% | 11.2% | 10.1% | +0.03 |
| 5 | 0.753 | 9.56 | 10.0% | 10.5% | 10.2% | -0.05 |
| 6 | 1.112 | 8.13 | 10.0% | 10.6% | 11.2% | +0.29 |
| 7 | 1.641 | 9.63 | 10.0% | 10.1% | 11.8% | +0.15 |
| 8 | 2.423 | 9.80 | 10.0% | 9.2% | 11.6% | +0.08 |
| 9 | 3.577 | 9.82 | 10.0% | 8.2% | 11.7% | +0.15 |

### ③ groups (halves by single-head FPR, per seed)

| all 10 heads | better half only | worse half only | damage from the worse half (all - better half) |
| --- | --- | --- | --- |
| 8.63 | 8.21 | 10.10 | +0.41 |
