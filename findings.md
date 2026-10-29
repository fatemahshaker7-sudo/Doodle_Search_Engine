# Findings

## Model choice
MobileNetV2 (ImageNet, `include_top=False`, average pooling): ~14 MB, fast on laptop CPU,
96x96 input (smallest allowed), 1280-d output vector.

## Quality
| Method | precision@5 | precision@25 |
|---|---|---|
| Raw pixels | | |
| Your model | | |
| Your model, inverted colors off | | |

## Threshold
threshold = ___ ; in-index queries wrongly rejected: ___ ; held-out wrongly accepted: ___

## 3 failures
1.
2.
3.

Data: Quick, Draw! dataset, made available by Google, Inc. under CC BY 4.0.
