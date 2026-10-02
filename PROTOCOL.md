# Main-text experiment protocol

This document defines the experiments included in the PURE v1.0.0 code
release. The scope is limited to the experiments reported in the main text.

## Canonical identifiers

- POI source outside China: Overture Places release 2024-12-18.0.
- China POI source: AMAP records represented with the same English template.
- POI text: A place of {category_leaf}, a type of {category_top}, named {name}.
- Text encoder: Qwen/Qwen3-Embedding-8B.
- Retained text dimensions: the first 512 dimensions of the 4,096-dimensional
  Matryoshka embedding, followed by L2 normalisation.
- Training records: 84,618,817 aligned AEF, POI-semantic and location triplets.
- Physical input: 64-dimensional AlphaEarth Foundations 2024 representation.
- PURE output: 128 dimensions.
- Checkpoint: checkpoints/PURE_epoch0010.pth.
- Checkpoint SHA256:
  6e1d51a795bafd0ece99f6494ca7e521b8614a953b4ba036905f7b6b4e611124.

## Representation model and training

The physical pathway uses the AEProjSH architecture with a gated residual
projection, hidden width 2,048 and output width 128. Geographic position is
encoded using degree-8 real spherical harmonics. The position branch has
hidden width 256 and a learnable scale initialised to 0.001. The POI text
pathway linearly projects the 512-dimensional text feature to the shared
128-dimensional space.

Training combines physical-to-POI contrastive alignment, centre-to-local-view
consistency and AEF relational distillation:

- image-image consistency weight: 0.2;
- text-alignment weight: 0.8;
- relational-distillation weight: 0.5;
- contrastive temperatures: 0.07;
- relational temperature: 0.1;
- relational sample size: at most 512 rows per batch.

The spatial sampler allocates each batch across approximately 0-10 km,
10-100 km, 100-1,000 km and global-other contexts with ratios
0.5/0.2/0.2/0.1. The exact regional weights and fallback rules are recorded
in code/configs/train_pure.yaml.

The main model was trained for 10 epochs with batch size 2,048, AdamW,
learning rate 1e-4, weight decay 1e-4, no learning-rate decay and seed 45.

## Global downstream benchmark

The benchmark contains GDP (2024), population (2024), nighttime lights
(2024), PM2.5 (2022) and daytime land-surface temperature (2024). Settlement
support uses GHS-SMOD R2023A V2.0, epoch 2025, classes 13, 21, 22, 23 and 30.

AEF and PURE pixels are overlap-area averaged to each target cell and then
L2-normalised. A separate MLP head is fitted for every country and task. The
head has one hidden layer of width 1,024, maximum 100 epochs, learning rate
0.001 and early-stopping patience 10. Population uses batch size 8,192; the
other tasks use batch size 128.

Targets are standardised using training rows only and predictions are
inverse-transformed before evaluation. Splits use 1-km cells in EPSG:6933.
The five downstream seeds are 42, 24, 7, 0 and 100.

The reporting cohort is the fixed intersection of 142 countries and
territories with more than 10,000 valid population cells and more than 1,000
valid cells for each other task. CHN, TWN, HKG and MAC are merged into the CHN
evaluation unit before splitting and standardisation.

## POI availability analysis

Country POI availability is the number of representation-training POIs
divided by WorldPop 2024 population and multiplied by 1,000. Each task uses
one ordinary least-squares model across all 142 countries:

delta R2 = intercept + beta * POIs per 1,000 people.

No country is removed as an outlier. Plot clipping does not affect fitting.
Countries receive equal weight in continent summaries.

## City POI-retention experiment

The 15 target cities are Bangkok, Chengdu, Osaka, Mumbai, Melbourne,
Johannesburg, Nairobi, Luanda, Lisbon, Paris, Manchester, Montreal,
Los Angeles, Rio de Janeiro and Buenos Aires.

City uses target-city POIs. Country uses all POIs in the containing country.
Country LOCO removes every target-city POI from that country pool. Each pool
uses nested 1%, 2%, 5%, 10%, 20%, 40%, 60%, 80% and 100% subsets generated
with seed 45. Percentages are relative to each pool and therefore do not imply
matched absolute POI counts.

Each representation is trained from scratch for 10 epochs with batch size
512, fixed learning rate 1e-4 and seed 45. Downstream evaluation uses the same
five tasks, target observations, spatial splits and five seeds as the global
benchmark.

## Cross-country transfer

Fifteen source-country representations are trained separately using all POIs
from Thailand, China, Japan, India, Australia, South Africa, Kenya, Angola,
Portugal, France, the United Kingdom, Canada, the United States, Brazil and
Argentina. Each frozen representation is evaluated in all 15 target cities,
forming 225 source-target combinations. The 15 diagonal combinations are
matched contexts; the other 210 are cross-country transfers. A new
target-city downstream head is fitted in every evaluation, so this is
representation transfer rather than zero-shot transfer of a prediction head.

## Transfer predictors

AEF similarity is calculated independently of POIs and downstream labels.
For every source country and target city, 5,000 unique valid native 100-m AEF
cells are sampled uniformly in equal-area space over the complete settlement
support. The mean 64-dimensional AEF vectors are L2-normalised before cosine
similarity is calculated.

Geographic distance is the great-circle distance from the source-country
capital to the centroid of the target city's predefined settlement support.
Target-city centroids are calculated in EPSG:6933 and transformed to
geographic coordinates. Pretoria is used for South Africa. Distance is
natural-log transformed.

AEF similarity and log distance are fitted in separate linear mixed-effects
models over the 210 off-diagonal transfers. Each model uses crossed random
intercepts for source country and target city. Predictor and response are
standardised, and fixed-effect significance uses a two-sided Wald test.

## Released embeddings

The global 128-dimensional PURE embeddings are available at:

- https://spacetimeai.com/pure/
- https://archive.org/details/poi-aligned-urban-embedding-v1-index

They are distributed as georeferenced, 8-bit-quantised GeoTIFF files by
country and first-level administrative region, with decoding parameters.
The original AMAP POI records are not redistributed.
