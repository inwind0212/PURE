# PURE: POI-aligned Urban Embedding

Code, checkpoint and result tables for the experiments reported in the main
text of *Urban functional knowledge transfer through global representations
amid uneven data* (Liu, Wang, Dong, Sha, Zeng, Deng, Feng, Wang and Cheng).

## Contents

- code/: data preparation, representation training, downstream evaluation and
  transfer experiments.
- checkpoints/: the epoch-10 PURE checkpoint.
- results/main/: the 142-country benchmark and the POI-availability analysis.
- results/transfer/: the POI-retention and 15-by-15 transfer results.
- figures/: the four main-text figures.
- PROTOCOL.md: the exact main-text experiment protocol.
- CITATION.cff: machine-readable citation metadata.

Supplementary-only analyses, source datasets and manuscript files are not
included.

## Released embeddings

Global PURE GeoTIFFs and decoding information are available from:

- https://spacetimeai.com/pure/
- https://archive.org/details/poi-aligned-urban-embedding-v1-index

## Environment

~~~bash
conda env create -f environment.yml
conda activate pure
~~~

Alternatively, install requirements.txt into a compatible Python 3.11 and
CUDA environment.

## Main training

~~~bash
export PURE_FEATURE_STORE=/path/to/pure_feature_store
export PURE_OUTPUT_DIR=/path/to/output
python code/training/train_global_aether.py   --cfg code/configs/train_pure.yaml   --store-dir "$PURE_FEATURE_STORE"
~~~

The large POI, AEF and downstream source datasets are not duplicated. Dataset
versions and experiment definitions are recorded in PROTOCOL.md.

## Transfer experiments

Set PURE_FEATURE_STORE, PURE_OUTPUT_DIR, PURE_TRANSFER_ROOT,
PURE_SETTLEMENT_SUPPORT_ROOT, PURE_CITY_RAW_ROOT and PURE_AE_CACHE_ROOT.
PURE_PYTHON is optional and defaults to the active interpreter.

~~~bash
python code/transfer/run_transfer_experiments.py --prepare
python code/transfer/run_transfer_experiments.py   --train-worker --worker-index 0 --workers 1 --physical-gpu 0
python code/transfer/run_transfer_experiments.py   --eval-worker --worker-index 0 --workers 1 --physical-gpu 0
python code/transfer/run_transfer_experiments.py --summarize
python code/transfer/analyze_transfer_associations.py
~~~

The last command computes geographic distances and fits the mixed-effects
models relating transfer gain to AEF similarity and distance.

## Figures and source tables

| Figure | Content | Source tables |
|---|---|---|
| figures/Figure1.png | PURE framework and transfer of urban functional knowledge to 15 target cities | results/main/global_five_task_summary.csv, results/main/poi_availability_country_poi_population.csv, results/transfer/poi_fraction_three_scopes_by_city.csv, results/transfer/country_transfer_pairs_with_predictors.csv |
| figures/Figure2.png | Country-level performance of AEF and PURE across five tasks | results/main/global_five_task_country_seed.csv |
| figures/Figure3.png | POI availability and the distribution of PURE gains | results/main/poi_availability_*.csv |
| figures/Figure4.png | Transferability of urban functional knowledge across geographic contexts | results/transfer/poi_fraction_three_scopes_mean.csv, results/transfer/country_transfer_15x15_delta_matrix.csv, results/transfer/country_transfer_pairs_with_predictors.csv, results/transfer/country_transfer_mixed_effects.json |

## Citation and licences

Citation metadata are provided in CITATION.cff.

- Code, model weights, configuration and environment files: MIT.
- PURE embeddings, results, figures and documentation: CC BY 4.0.
- Third-party datasets retain their original licences.

See LICENSE, LICENSE-CODE and LICENSE-DATA.
