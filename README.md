# ML4MSD Group Project: coGN & coNGN OOD Evaluation

Evaluating graph neural networks for 2D material property prediction under out-of-distribution conditions using [MatFold](https://github.com/schindlerlab/MatFold) splits.

**Course:** ML4MSD @ Northeastern University  
**Author:** Ardavan Mehdizadeh  
**Date:** December 2025

## Background

Standard benchmarks like Matbench use random splits, which leak information between train/test sets (similar compositions appear in both). MatFold provides systematic OOD splits that actually test generalization.

## Models

| Model | Description |
|-------|-------------|
| **coGN** | Crystal graph network — atoms as nodes, bonds as edges |
| **coNGN** | Adds line graph to capture bond angles |

## Dataset

- **matbench_jdft2d** — 2D materials exfoliation energies (636 samples)
- 70/20/10 train/val/test split

## Results

| Split | coGN (meV/atom) | coNGN (meV/atom) | Winner |
|-------|-----------------|------------------|--------|
| composition | 27.2 | 30.6 | coGN |
| chemsys | 49.6 | 45.1 | coNGN |
| sgnum | 52.9 | 46.6 | coNGN |
| periodictablegroups | 52.4 | 53.7 | coGN |
| pointgroup | 59.8 | 51.6 | coNGN |
| elements | 60.7 | 91.0 | coGN |
| **Average** | **50.4** | **53.1** | **coGN** |

### Key findings

1. **coGN is more robust overall** (lower average MAE)
2. **coNGN wins on structural OOD** — 9-14% better on pointgroup/sgnum/chemsys
3. **coNGN fails hard on unseen elements** — line graph overfits to element-specific bond patterns
4. **Pick your model based on use case:** coNGN for new structures, coGN for new elements

## Structure

```
├── scripts/
│   ├── test_coGN_matfold.py      # main coGN script
│   ├── test_coNGN_matfold.py     # main coNGN script
│   └── run_coGN_all_splits.sh    # batch submission
├── results/
│   ├── coGN_matfold/             # JSONs, CSVs, saved models
│   └── coNGN_matfold/
├── environment.yml               # conda env spec
└── README.md
```

## Saved Models

Each trained model is saved in TensorFlow SavedModel format:
```
coGN_chemsys_model/
├── fingerprint.pb              # model hash
├── keras_metadata.pb           # architecture metadata
├── saved_model.pb              # model graph
└── variables/
    ├── variables.data-00000-of-00001   # trained weights
    └── variables.index
```

### Loading a model
```python
import tensorflow as tf

model = tf.keras.models.load_model('results/coGN_matfold/coGN_chemsys_model')

# check inputs
print([inp.name for inp in model.inputs])
# ['offset', 'atomic_number', 'multiplicity', 'edge_indices']
```

### Model sizes

| Model | Size per split |
|-------|----------------|
| coGN  | ~12 MB |
| coNGN | ~40 MB |

coNGN is larger due to the additional line graph layers for bond angle encoding.


## Setup

```bash
conda env create -f environment.yml
conda activate cogn_env
```

## Usage

Single split:
```bash
# edit SPLIT_TYPE in script, then:
python scripts/test_coGN_matfold.py
```

All splits (HPC):
```bash
bash scripts/run_coGN_all_splits.sh
squeue -u $USER  # monitor
```

Load saved model:
```python
import tensorflow as tf
model = tf.keras.models.load_model('results/coGN_matfold/coGN_chemsys_model')
```

## Split Types

| Split | Tests |
|-------|-------|
| composition | same elements, different ratios |
| chemsys | entirely different element combos |
| elements | completely unseen elements |
| periodictablegroups | cross-group generalization |
| pointgroup | different symmetries |
| sgnum | different space groups |

## Training Config

- 50 epochs, batch 32, Adam @ 1e-3
- KNN graph with k=12
- StandardScaler on targets

## Acknowledgments

- Prof. Peter Schindler (course instructor, MatFold developer)