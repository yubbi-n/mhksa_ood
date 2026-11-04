# KPCA vs Ours — 데이터셋 3개 (CE)

FPR95 (%), 낮을수록 좋음. seed 5개 평균.
Ours는 세 데이터셋 모두 **같은 고정 규칙**: head 10개, γ = median γ × 0.3 ~ 10, PCA 비율 = √(KPCA 비율). test OOD로 고른 값 없음.

| group | method | CIFAR-10 | CIFAR-100 | ImageNet |
| --- | --- | --- | --- | --- |
| KPCA | median γ | 40.28 | 73.33 | 69.22 |
| KPCA | 논문의 튜닝 γ | 28.63 | 없음 | 48.60 |
| Ours | MHKSA | **27.39** | **66.75** | **48.43** |
| Ours | Ensemble | 29.04 | 68.32 | 60.00 |

## 결론
1. **세 데이터셋 모두 median KPCA보다 좋다**: CIFAR-10 −12.9, CIFAR-100 −6.6, ImageNet −20.8.
2. **γ 튜닝 없이 튜닝된 KPCA와 같거나 더 좋다**: CIFAR-10 27.39 vs 28.63, ImageNet 48.43 vs 48.60.
3. **범위를 더 넓혀도 그대로다**: CIFAR-10 ×30 27.88, ImageNet ×20 48.02 → 범위만 충분히 넓으면 됨.
4. **MHKSA가 Ensemble보다 낫다**: 특히 ImageNet (48.43 vs 60.00).

※ 논문의 튜닝 γ: CIFAR-10 γ=2 (논문 코드 설정으로 우리가 실행), ImageNet γ=3 (논문 Table 1 수치, 우리 재현 48.63). 논문에 CIFAR-100 결과 없음.
