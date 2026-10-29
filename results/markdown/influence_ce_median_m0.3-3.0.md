# influence: ce   run=CIFAR10-R18-ce-evr0.8-mh0.9_0.9-M2048-m0.3_3.0log   H=10   seeds=[0, 1, 2, 3, 4]

## single heads (FPR95, AVG over OOD sets) and rank (0 = best)

| head | gamma | FPR95 | rank |
| --- | --- | --- | --- |
| 0 | 0.128 | 38.50 | 6 |
| 1 | 0.165 | 36.82 | 3 |
| 2 | 0.213 | 35.84 | 2 |
| 3 | 0.275 | 41.03 | 9 |
| 4 | 0.355 | 38.74 | 7 |
| 5 | 0.459 | 38.89 | 8 |
| 6 | 0.593 | 37.82 | 5 |
| 7 | 0.765 | 37.72 | 4 |
| 8 | 0.989 | 34.03 | 1 |
| 9 | 1.277 | 32.50 | 0 |

sanity: refitted stage 2 on all heads vs checkpoint MHKSA, max |score diff| = 7.15e-07

## rule: MHKSA (stage 2 refitted per head subset)   (all 10 heads: FPR95 35.39)

### ① influence vs contribution, and ② leave-one-head-out

(shares: stage-1 terms e_h^2 only; FPR columns: full MHKSA with the refitted stage 2)

| head | gamma | single FPR | weight on ID | share of OOD-ID gap (signal) | share of ID variance (noise) | FPR change if removed (- = harmful head) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.128 | 38.50 | 3.3% | 2.4% | 0.5% | -0.35 |
| 1 | 0.165 | 36.82 | 4.1% | 3.0% | 0.8% | -0.65 |
| 2 | 0.213 | 35.84 | 5.2% | 4.0% | 1.3% | -0.45 |
| 3 | 0.275 | 41.03 | 6.5% | 5.0% | 2.1% | -0.67 |
| 4 | 0.355 | 38.74 | 8.0% | 6.6% | 3.3% | -0.74 |
| 5 | 0.459 | 38.89 | 10.0% | 8.7% | 5.4% | -0.54 |
| 6 | 0.593 | 37.82 | 12.0% | 11.2% | 8.7% | -0.27 |
| 7 | 0.765 | 37.72 | 14.4% | 15.1% | 14.8% | +0.05 |
| 8 | 0.989 | 34.03 | 17.1% | 20.0% | 25.3% | +2.27 |
| 9 | 1.277 | 32.50 | 19.5% | 24.0% | 37.7% | +3.62 |

### ③ groups (halves by single-head FPR, per seed)

| all 10 heads | better half only | worse half only | damage from the worse half (all - better half) |
| --- | --- | --- | --- |
| 35.39 | 33.32 | 39.36 | +2.08 |

## rule: z-score mean (ensemble)   (all 10 heads: FPR95 32.38)

### ① influence vs contribution, and ② leave-one-head-out

| head | gamma | single FPR | weight on ID | share of OOD-ID gap (signal) | share of ID variance (noise) | FPR change if removed (- = harmful head) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.128 | 38.50 | 10.0% | 10.3% | 9.4% | +0.45 |
| 1 | 0.165 | 36.82 | 10.0% | 10.5% | 9.5% | +0.34 |
| 2 | 0.213 | 35.84 | 10.0% | 10.3% | 9.7% | +0.35 |
| 3 | 0.275 | 41.03 | 10.0% | 9.2% | 9.8% | -0.49 |
| 4 | 0.355 | 38.74 | 10.0% | 9.6% | 10.0% | -0.34 |
| 5 | 0.459 | 38.89 | 10.0% | 9.4% | 10.2% | -0.20 |
| 6 | 0.593 | 37.82 | 10.0% | 9.7% | 10.3% | +0.08 |
| 7 | 0.765 | 37.72 | 10.0% | 9.8% | 10.2% | +0.20 |
| 8 | 0.989 | 34.03 | 10.0% | 10.4% | 10.3% | +0.49 |
| 9 | 1.277 | 32.50 | 10.0% | 10.8% | 10.6% | +0.77 |

### ③ groups (halves by single-head FPR, per seed)

| all 10 heads | better half only | worse half only | damage from the worse half (all - better half) |
| --- | --- | --- | --- |
| 32.38 | 30.90 | 35.03 | +1.47 |
