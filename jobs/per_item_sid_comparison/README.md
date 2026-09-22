# Per-Item Signal Comparison

This workflow compares per-item length signals across datasets, SID methods, and direct/inverse directions.

## Submit

```bash
bash jobs/per_item_sid_comparison/submit_per_item_sid_comparison.sh
```

Defaults:

- Datasets: `beauty`, `toys`, `games`
- SID methods: `rkmeans`
- Maximum SID length: `10`
- Directions: `direct` and `inverse`
- Signals: all eight supported per-item methods

Edit the arrays in `submit_per_item_sid_comparison.sh` to select datasets, methods, directions, or signals. For example:

```bash
DATASETS=(beauty toys)
SID_METHODS=(rkmeans rvq)
PER_ITEM_METHODS=(interaction_count ppmi item2vec)
DIRECTIONS=(direct inverse)
```

## Job Graph

The submission creates:

1. One length-generation and length-analysis job per direction.
2. One signal-specific SID/TIGER job per dataset, method, signal, and direction.
3. One final metrics aggregation job per dataset dependent on all model jobs for that dataset.

Signal jobs reuse a shared tokenizer checkpoint using a filesystem lock.

## Outputs

Results are stored under:

```text
logs/{dataset}/per_item_comparison/{run_id}/
```

The directory contains per-direction length reports, signal-specific SID and TIGER runs, and `final_signal_metrics.csv`.
