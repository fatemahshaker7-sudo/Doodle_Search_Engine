# DoodleDex findings

Data: Quick, Draw! dataset, made available by Google, Inc. under the Creative Commons Attribution 4.0 International license.

Index: 10 categories (cat, dog, sun, smiley face, cup, bucket, apple, house, car, fish), 450 gallery doodles each (4,500 total) and 50 queries each (500 total). Held-out: cell phone (50 doodles). Seed 0.

## Model choice

I used MobileNetV2 (ImageNet weights, no top layer, average pooling). It is small (about 14 MB), fast on a laptop CPU, accepts inputs as small as 96×96 (close to the 28×28 doodles' native resolution, so little upscaling), and outputs a 1280-dimensional vector. No training or fine-tuning was done.

## Quality table

| Method | precision@5 | precision@25 |
|---|---|---|
| Raw pixels | 0.683 | 0.584 |
| Your model (MobileNetV2, inverted) | 0.850  | 0.802 |
| Your model, inverted colors off | 0.828 | 0.779 |

The model beats raw pixels by about 19 points at p@5 and 24 points at p@25. Inverting the colors (black strokes on white) helps slightly (+1.4 points at p@5, +1.6 at p@25), consistent with the idea that this looks more like the photos the model was trained on. The gap is small, so it is a modest effect, not a decisive one.

Caveat: these numbers use clean Quick, Draw! queries that skip `preprocess_query`. Real canvas drawings and uploads go through extra preprocessing, so live quality may differ.

## Threshold

Calibrated threshold: **0.842** (saved in `index/config.json` with `top_k` 25). It maximizes balanced accuracy between in-index queries and held-out doodles. Histogram: `threshold_hist.png`.

- In-index queries wrongly rejected: **155/500 (31%)**
- Held-out (cell phone) wrongly accepted: **4/50 (8%)**

The histograms overlap heavily: many ordinary doodles score below 0.842, while some phones score above it. A phone is a plain rectangle and resembles house, cup or bucket outlines. This is a finding about the model's limits, not a bug. With only one held-out category, the threshold may also be tuned narrowly to phones. Using more held-out categories would make it sturdier.

## Visual check (Task 4, step 6)

Figure: `good_bad_results.png`.

Good queries:
1. (cat, p@25=1) - cat
2. (fish, p@25=1) - fish
3. (sun, p@25=1) - sun

Bad queries (one line on why each failed):
1. (fish, p@25=0) - most common wrong category: car
2. (dog, p@25=0) - most common wrong category: fish
3. (sun, p@25=0) - most common wrong category: apple

## Preprocessing check (Task 3)

Figure: `preprocess_check.png` (Paint drawing, phone photo). Notes: 

## Task 6 failures

| # | Input | What came back | Cause (preprocessing gap / lookalike shapes / threshold too high or low / model limits) |
|---|---|---|---|
| 1 | fish | car | lookalike shapes |
| 2 | dog | fish | model limits |
| 3 | sun | apple | because they are both circles (lookalike shapes) | 

## Other notes

- Labels are used only in `evaluate.py`. The app never shows category names.
- The camera input was out of scope for this submission; the app has Draw and Upload tabs only.
