# influence: supcon   run=CIFAR10-R18-supcon-evr0.7-mh0.84_0.84-M2048-m0.3_3.0log   H=10   seeds=[0, 1, 2, 3, 4]

## single heads (FPR95, AVG over OOD sets) and rank (0 = best)

| head | gamma | FPR95 | rank |
| --- | --- | --- | --- |
| 0 | 0.107 | 14.22 | 6 |
| 1 | 0.139 | 15.48 | 8 |
| 2 | 0.179 | 15.48 | 9 |
| 3 | 0.231 | 15.22 | 7 |
| 4 | 0.299 | 13.13 | 5 |
| 5 | 0.386 | 9.87 | 4 |
| 6 | 0.498 | 9.67 | 3 |
| 7 | 0.643 | 9.67 | 2 |
| 8 | 0.831 | 9.52 | 1 |
| 9 | 1.073 | 7.98 | 0 |

sanity: refitted stage 2 on all heads vs checkpoint MHKSA, max |score diff| = 7.75e-07

## rule: MHKSA (stage 2 refitted per head subset)   (all 10 heads: FPR95 9.42)

### ① influence vs contribution, and ② leave-one-head-out

(shares: stage-1 terms e_h^2 only; FPR columns: full MHKSA with the refitted stage 2)

| head | gamma | single FPR | weight on ID | share of OOD-ID gap (signal) | share of ID variance (noise) | FPR change if removed (- = harmful head) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.107 | 14.22 | 3.3% | 3.6% | 0.8% | -0.03 |
| 1 | 0.139 | 15.48 | 4.1% | 4.5% | 1.2% | -0.03 |
| 2 | 0.179 | 15.48 | 5.1% | 5.6% | 1.9% | -0.05 |
| 3 | 0.231 | 15.22 | 6.5% | 6.9% | 3.0% | -0.07 |
| 4 | 0.299 | 13.13 | 8.1% | 8.4% | 4.8% | -0.01 |
| 5 | 0.386 | 9.87 | 10.1% | 10.2% | 7.4% | -0.03 |
| 6 | 0.498 | 9.67 | 12.1% | 12.0% | 10.8% | -0.00 |
| 7 | 0.643 | 9.67 | 14.4% | 14.1% | 16.0% | -0.01 |
| 8 | 0.831 | 9.52 | 17.0% | 16.3% | 22.9% | +0.04 |
| 9 | 1.073 | 7.98 | 19.3% | 18.4% | 31.3% | +0.18 |

### ③ groups (halves by single-head FPR, per seed)

| all 10 heads | better half only | worse half only | damage from the worse half (all - better half) |
| --- | --- | --- | --- |
| 9.42 | 9.26 | 15.02 | +0.16 |

## rule: z-score mean (ensemble)   (all 10 heads: FPR95 9.68)

### ① influence vs contribution, and ② leave-one-head-out

| head | gamma | single FPR | weight on ID | share of OOD-ID gap (signal) | share of ID variance (noise) | FPR change if removed (- = harmful head) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.107 | 14.22 | 10.0% | 9.7% | 8.8% | -0.18 |
| 1 | 0.139 | 15.48 | 10.0% | 9.4% | 8.6% | -0.18 |
| 2 | 0.179 | 15.48 | 10.0% | 9.4% | 8.7% | -0.18 |
| 3 | 0.231 | 15.22 | 10.0% | 9.0% | 8.7% | -0.13 |
| 4 | 0.299 | 13.13 | 10.0% | 9.2% | 9.0% | -0.10 |
| 5 | 0.386 | 9.87 | 10.0% | 11.5% | 11.0% | +0.07 |
| 6 | 0.498 | 9.67 | 10.0% | 11.1% | 11.0% | +0.14 |
| 7 | 0.643 | 9.67 | 10.0% | 10.5% | 11.0% | +0.08 |
| 8 | 0.831 | 9.52 | 10.0% | 9.9% | 11.1% | +0.16 |
| 9 | 1.073 | 7.98 | 10.0% | 10.4% | 12.1% | +0.84 |

### ③ groups (halves by single-head FPR, per seed)

| all 10 heads | better half only | worse half only | damage from the worse half (all - better half) |
| --- | --- | --- | --- |
| 9.68 | 8.59 | 12.81 | +1.09 |
