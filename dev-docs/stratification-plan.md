# Stratification Plan

## Context
- Current dataset uses one image per VIN.
- Full VIN labels are unique, so stratified splitting on VIN fails.
- For now, stratification is disabled and splits are random.

## Goal
Enable stratified sampling without violating the one-image-per-VIN rule by stratifying on a coarser, repeatable group label.

## Options
1) WMI (first 3 VIN characters)
   - Coarse manufacturer grouping.
   - Likely to have repeats even with one image per VIN.

2) VIN prefix (first 8 VIN characters)
   - Finer grouping than WMI.
   - May still have repeats depending on dataset diversity.

3) Source/batch group
   - If capture batch or folder name encodes acquisition session.
   - Useful for balancing across collection sources.

## Proposed Implementation
- Add a `stratify_mode` parameter with values:
  - `none` (default for idempotent one-image-per-VIN datasets)
  - `wmi`
  - `vin_prefix`
  - `source`
- Add a `stratify_prefix_len` parameter for prefix-based grouping (default 3 for WMI).
- Create `stratify_labels` from the chosen mode and pass to `train_test_split`.
- Validate: if any stratify group has < 2 samples, fallback to `none` with a warning.

## Validation Steps
- Print group counts and min frequency before splitting.
- Confirm train/val/test group distribution is stable across runs (fixed seed).
- Confirm no image appears in more than one split.

## Notes
- Full VIN stratification is incompatible with one-image-per-VIN datasets.
- Group-based stratification preserves idempotency while maintaining balance.
