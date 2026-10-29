# influence: ce   run=CIFAR10-R18-ce-evr0.8-mh0.9_0.9-M2048-m0.3_10log   H=10   seeds=[0, 1, 2, 3, 4]

## single heads (FPR95, AVG over OOD sets) and rank (0 = best)

| head | gamma | FPR95 | rank |
| --- | --- | --- | --- |
| 0 | 0.128 | 38.50 | 7 |
| 1 | 0.189 | 36.59 | 5 |
| 2 | 0.278 | 40.28 | 9 |
| 3 | 0.411 | 38.83 | 8 |
| 4 | 0.607 | 37.87 | 6 |
| 5 | 0.896 | 34.29 | 4 |
| 6 | 1.322 | 31.89 | 3 |
| 7 | 1.953 | 27.93 | 0 |
| 8 | 2.883 | 28.13 | 1 |
| 9 | 4.256 | 28.77 | 2 |

sanity: refitted stage 2 on all heads vs checkpoint MHKSA, max |score diff| = 8.34e-07

## rule: MHKSA (stage 2 refitted per head subset)   (all 10 heads: FPR95 27.39)

### ① influence vs contribution, and ② leave-one-head-out

(shares: stage-1 terms e_h^2 only; FPR columns: full MHKSA with the refitted stage 2)

| head | gamma | single FPR | weight on ID | share of OOD-ID gap (signal) | share of ID variance (noise) | FPR change if removed (- = harmful head) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.128 | 38.50 | 2.2% | 1.3% | 0.2% | -0.01 |
| 1 | 0.189 | 36.59 | 3.1% | 1.9% | 0.3% | -0.01 |
| 2 | 0.278 | 40.28 | 4.4% | 2.8% | 0.6% | -0.03 |
| 3 | 0.411 | 38.83 | 6.0% | 4.3% | 1.2% | +0.07 |
| 4 | 0.607 | 37.87 | 8.1% | 6.4% | 2.6% | +0.10 |
| 5 | 0.896 | 34.29 | 10.7% | 10.0% | 6.0% | +0.14 |
| 6 | 1.322 | 31.89 | 13.2% | 13.8% | 11.3% | +0.20 |
| 7 | 1.953 | 27.93 | 15.5% | 17.5% | 18.2% | +0.11 |
| 8 | 2.883 | 28.13 | 17.6% | 20.8% | 27.0% | +0.33 |
| 9 | 4.256 | 28.77 | 19.2% | 21.2% | 32.6% | +1.27 |

### ③ groups (halves by single-head FPR, per seed)

| all 10 heads | better half only | worse half only | damage from the worse half (all - better half) |
| --- | --- | --- | --- |
| 27.39 | 27.90 | 40.97 | -0.50 |

## rule: z-score mean (ensemble)   (all 10 heads: FPR95 29.04)

### ① influence vs contribution, and ② leave-one-head-out

| head | gamma | single FPR | weight on ID | share of OOD-ID gap (signal) | share of ID variance (noise) | FPR change if removed (- = harmful head) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.128 | 38.50 | 10.0% | 9.9% | 8.8% | +0.29 |
| 1 | 0.189 | 36.59 | 10.0% | 10.1% | 8.9% | +0.31 |
| 2 | 0.278 | 40.28 | 10.0% | 9.0% | 9.2% | -0.55 |
| 3 | 0.411 | 38.83 | 10.0% | 9.1% | 9.5% | -0.39 |
| 4 | 0.607 | 37.87 | 10.0% | 9.3% | 9.6% | -0.31 |
| 5 | 0.896 | 34.29 | 10.0% | 10.0% | 9.6% | -0.15 |
| 6 | 1.322 | 31.89 | 10.0% | 10.5% | 10.1% | -0.02 |
| 7 | 1.953 | 27.93 | 10.0% | 10.8% | 10.9% | +0.39 |
| 8 | 2.883 | 28.13 | 10.0% | 10.8% | 11.3% | +0.53 |
| 9 | 4.256 | 28.77 | 10.0% | 10.6% | 12.1% | +0.65 |

### ③ groups (halves by single-head FPR, per seed)

| all 10 heads | better half only | worse half only | damage from the worse half (all - better half) |
| --- | --- | --- | --- |
| 29.04 | 28.12 | 33.99 | +0.92 |
