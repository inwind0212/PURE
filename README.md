# PURE: POI-aligned Urban Embedding

This is the clean v1.0.0 release of the code, checkpoint and result tables for
the experiments reported in the main text of *Urban Functional Knowledge
Transfer through Global Representations amid Uneven Data*.

## Contents

- code/: training, evaluation, transfer and figure-generation code.
- checkpoints/: the epoch-10 PURE checkpoint.
- results/main/: the 142-country benchmark and Figure 2 inputs.
- results/transfer/: the POI-retention and 15-by-15 transfer results.
- figures/: one PNG preview per main figure and the required Figure 3 artwork.
- audit/: machine-readable provenance, correctness and validation records.
- manifest/: file sizes and SHA256 checksums.
- PROTOCOL.md: the exact main-text experiment protocol.
- CITATION.cff: machine-readable citation metadata.

Historical experiments, supplementary-only analyses, source datasets and
manuscript files are intentionally excluded.

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
versions and experiment definitions are recorded in PROTOCOL.md and
audit/PROVENANCE.json.

## Main figures

Figures 2 and 3 regenerate from released result tables and assets:

~~~bash
python code/figures/plot_figure2.py
python code/figures/plot_figure3.py
~~~

Figure 1 additionally requires a Natural Earth-compatible country boundary
file:

~~~bash
python code/figures/plot_figure1.py   --boundaries /path/to/ne_10m_admin_0_countries.shp
~~~

Generated files are written under outputs/, which is excluded from version
control. The retained PNG files under figures/ are publication previews.

## Transfer experiments

Set PURE_FEATURE_STORE, PURE_OUTPUT_DIR, PURE_TRANSFER_ROOT,
PURE_SETTLEMENT_SUPPORT_ROOT, PURE_CITY_RAW_ROOT and PURE_AE_CACHE_ROOT.
PURE_PYTHON is optional and defaults to the active interpreter.

~~~bash
python code/transfer/run_transfer_experiments.py --prepare
python code/transfer/run_transfer_experiments.py   --train-worker --worker-index 0 --workers 1 --physical-gpu 0
python code/transfer/run_transfer_experiments.py   --eval-worker --worker-index 0 --workers 1 --physical-gpu 0
python code/transfer/run_transfer_experiments.py --summarize
~~~

## Verification

The release was checked by compiling all Python sources, validating the
checksum manifest and regenerating Figures 1-3 in an isolated directory.
Details are in audit/VALIDATION.md.

## Citation and licences

Citation metadata are provided in CITATION.cff.

- Code, model weights, configuration and environment files: MIT.
- PURE embeddings, results, figures and documentation: CC BY 4.0.
- Third-party datasets retain their original licences.

See LICENSE, LICENSE-CODE and LICENSE-DATA.
