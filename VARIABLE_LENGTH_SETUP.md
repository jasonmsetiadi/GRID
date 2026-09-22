# Variable-Length Semantic IDs

GRID supports two independent ways to choose the effective length of a semantic ID:

- **Content-based selection** derives the length from the item's quantization behavior.
- **Per-item selection** assigns an exact length to each item using a dataset or collaborative signal.

Both modes quantize items up to a configured maximum length and produce the same structured output.

## Configuration

The selection settings are passed to the semantic-ID model through the SID experiment configurations:

```yaml
model:
  variable_length: true
  length_selection_mode: content_based
  length_selection_method: residual_threshold
  min_sid_length: 1
  max_sid_length: 10
  residual_threshold: 0.1
```

The two top-level modes are:

```text
content_based
per_item
```

`variable_length: false` preserves the existing fixed-length behavior and ignores length selection during inference.

## Content-Based Selection

Content-based selection uses information produced by the quantizer for each item. The current implementation supports:

```yaml
length_selection_mode: content_based
length_selection_method: residual_threshold
```

### `residual_threshold`

After each quantization layer, GRID measures the remaining residual norm. The selected length is the first layer whose residual is at or below `residual_threshold`.

If no layer reaches the threshold, `max_sid_length` is used. `min_sid_length` is always respected.

Conceptually:

```text
for length in min_length ... max_length:
    if residual_norm[length] <= residual_threshold:
        select length
        stop
```

This mode requires residual tracking during SID inference.

### Future Content-Based Methods

The selector interface is designed to support additional content criteria without changing downstream data handling. Candidates include:

- `reconstruction_error`: choose the shortest prefix meeting a reconstruction-error target.
- `marginal_gain`: stop when another layer provides insufficient improvement.
- `rate_distortion`: minimize reconstruction error plus a length penalty.

These methods are not currently implemented.

## Per-Item Selection

Per-item selection assigns an exact length to every item. The selected length is not modified by the residual threshold.

```yaml
length_selection_mode: per_item
length_selection_method: interaction_count
min_sid_length: 1
max_sid_length: 10
```

The pipeline generates an internal length tensor indexed by `item_id` and applies:

```python
lengths = item_lengths[item_ids]
```

The generated lengths must fall within `min_sid_length` and `max_sid_length`. Invalid item IDs or lengths fail explicitly.

### Available Per-Item Methods

#### `interaction_count`

Assigns lengths from item frequency in the training sequences. With `direct` direction, more interactions produce longer IDs. With `inverse`, more interactions produce shorter IDs.

#### `cooccurrence`

Scores items by the number of other unique items appearing alongside them in training sequences.

#### `ppmi`

Uses positive pointwise mutual information between item pairs. This favors unusually strong associations rather than raw popularity.

#### `neighborhood_entropy`

Uses the entropy of each item's co-occurrence neighborhood. Items with more diverse neighborhoods receive different lengths from items with concentrated neighborhoods.

#### `graph_centrality`

Uses weighted item-graph degree based on co-occurrence edges.

#### `item2vec`

Trains a small item-context embedding model and maps item embedding norms to lengths.

#### `bpr`

Trains a lightweight Bayesian Personalized Ranking model and maps item embedding norms to lengths.

#### `lightgcn`

Trains a lightweight graph collaborative-filtering model with graph propagation and BPR-style updates, then maps propagated item embedding norms to lengths.

## Direction and Length Mapping

Per-item signal scores are normalized into the configured length range:

```yaml
LENGTH_DIRECTION=direct
```

- `direct`: larger signal produces a longer SID.
- `inverse`: larger signal produces a shorter SID.

The mapping is deterministic for a given training dataset, method, direction, and length range.

## Pipeline Usage

### Content-Based Run

```bash
LENGTH_SELECTION_MODE=content_based \
LENGTH_SELECTION_METHOD=residual_threshold \
RESIDUAL_THRESHOLD=0.1 \
SID_HIERARCHIES=10 \
bash jobs/run_full_pipeline.sh
```

### Per-Item Run

```bash
LENGTH_SELECTION_MODE=per_item \
LENGTH_SELECTION_METHOD=interaction_count \
LENGTH_DIRECTION=direct \
SID_HIERARCHIES=10 \
bash jobs/run_full_pipeline.sh
```

The pipeline generates the per-item length artifact internally under the run's SID log directory. It is an implementation artifact, not a separate user-facing selection method.

## Output Contract

Variable-length SID inference writes a structured artifact containing:

```text
codes:          [num_items, max_length]
lengths:        [num_items]
residual_norms: [num_items, max_length]
item_ids:       [num_items]
```

Only `codes[item_id, :lengths[item_id]]` is the semantic ID. Padding after the selected length is not part of the ID.

The same `codes` and `lengths` representation is consumed by preprocessing and TIGER. Content-based and per-item selection therefore differ only in how `lengths` is produced.

## SID Analysis

Generated SID artifacts can be compared before running TIGER:

```bash
python -m src.analyze_variable_length_sids \
  --artifact interaction_count=logs/beauty/rkmeans/per_item_interaction_count_direct/inference/runs/<run>/pickle/merged_predictions_tensor.pt \
  --artifact ppmi=logs/beauty/rkmeans/per_item_ppmi_direct/inference/runs/<run>/pickle/merged_predictions_tensor.pt \
  --artifact residual=logs/beauty/rkmeans/content_residual_threshold/inference/runs/<run>/pickle/merged_predictions_tensor.pt \
  --max-length 10 \
  --output-dir logs/beauty/sid_analysis
```

The analyzer writes:

- `sid_summary.csv`: length distributions, token savings, unique IDs, collisions, and selected residual statistics.
- `sid_pairwise.csv`: length agreement, correlation, effective SID agreement, and top-length-item overlap for every artifact pair.

Comparisons use `codes[item_id, :lengths[item_id]]`; padded codes are never treated as part of an SID.

## Comparison

| Property | Content-based | Per-item |
| --- | --- | --- |
| Length source | Quantization or reconstruction behavior | Training signal assigned to each item |
| Current method | `residual_threshold` | Eight signal methods |
| Uses residual threshold | Yes, for the current method | No |
| Length output | Item-specific | Item-specific |
| Requires training data at inference setup | No | Yes, to generate the signal artifact |
| Exact requested length | No, criterion determines it | Yes |

## Validation

The selector tests cover:

- Residual threshold selection.
- Minimum and maximum length enforcement.
- Exact per-item lengths.
- Per-item selection ignoring residual thresholds.
- Invalid selection methods and length ranges.

The local environment can run syntax and shell checks, but full runtime tests require the project environment with PyTorch, TensorFlow, and pytest installed.
