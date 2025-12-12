#!/bin/bash
# Run coNGN on all MatFold split types
# Usage: bash scripts/run_coNGN_all_splits.sh

cd /projects/d2r2/ardavan/ml4msd_group_project

# Skip chemsys since it's already done
SPLITS=("composition" "elements" "periodictablegroups" "pointgroup" "sgnum")

for split in "${SPLITS[@]}"; do
    echo "============================================"
    echo "Submitting coNGN job for split: $split"
    echo "============================================"
    
    # Create a temporary script with the split type
    cat > scripts/temp_coNGN_${split}.py << 'SCRIPT_END'
"""
coNGN (Nested) with MatFold Split
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time
import json
import tensorflow as tf
import networkx as nx
from pymatgen.core import Structure

from kgcnn.literature.coGN import make_model, model_default_nested
from kgcnn.crystal.preprocessor import KNNAsymmetricUnitCell
from kgcnn.data.transform.scaler.standard import StandardScaler

np.random.seed(42)
tf.random.set_seed(42)

SPLIT_TYPE = "SPLIT_PLACEHOLDER"
SPLIT_RATIO = "0.7-0.2-0.1"
SPLITS_DIR = Path("/projects/d2r2/ardavan/ml4msd_group_project/data/Splits-Yue")
RESULTS_DIR = Path("/projects/d2r2/ardavan/ml4msd_group_project/results/coNGN_matfold")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

base_name = f"mf.index-val_{SPLIT_TYPE}-test_{SPLIT_RATIO}"
train_file = SPLITS_DIR / f"{base_name}.train.csv"
val_file = SPLITS_DIR / f"{base_name}.validation.csv"
test_file = SPLITS_DIR / f"{base_name}.test.csv"

print("="*70)
print(f"coNGN with MatFold Split: {SPLIT_TYPE}")
print("="*70)

def load_split_file(filepath):
    df = pd.read_csv(filepath)
    structures = [Structure.from_dict(eval(s)) for s in df['structure']]
    target_col = [c for c in df.columns if c not in ['mat_index', 'structure']][0]
    return structures, df[target_col].values, target_col

print("Loading splits...")
train_structures, train_targets, target_col = load_split_file(train_file)
val_structures, val_targets, _ = load_split_file(val_file)
test_structures, test_targets, _ = load_split_file(test_file)
print(f"Train: {len(train_structures)}, Val: {len(val_structures)}, Test: {len(test_structures)}")

preprocessor = KNNAsymmetricUnitCell(k=12)

def compute_line_graph_indices(edge_indices):
    if len(edge_indices) == 0:
        return np.zeros((0, 2), dtype="int64")
    targets = edge_indices[:, 1]
    sources = edge_indices[:, 0]
    adjacency = (targets[:, None] == sources[None, :])
    edge_idx_1, edge_idx_2 = np.where(adjacency)
    return np.stack([edge_idx_1, edge_idx_2], axis=1).astype(np.int64)

def graph_to_tensor_data(nx_graph):
    nodes_z = [nx_graph.nodes[n].get('atomic_number', 0) for n in range(len(nx_graph.nodes))]
    nodes_mult = [nx_graph.nodes[n].get('multiplicity', 1) for n in range(len(nx_graph.nodes))]
    edges_u, edges_v, edges_offset = [], [], []
    for u, v, k, data in nx_graph.edges(keys=True, data=True):
        edges_u.append(u)
        edges_v.append(v)
        edges_offset.append(data.get('offset', [0, 0, 0]))
    edge_indices = np.array([edges_u, edges_v], dtype="int64").T
    line_graph_indices = compute_line_graph_indices(edge_indices)
    return {
        'atomic_number': np.array(nodes_z, dtype="int32"),
        'multiplicity': np.array(nodes_mult, dtype="int32"),
        'edge_indices': edge_indices,
        'line_graph_edge_indices': line_graph_indices,
        'offset': np.array(edges_offset, dtype="float32")
    }

def convert_structures(structures):
    graphs, valid_idx = [], []
    for i, struct in enumerate(structures):
        try:
            g = preprocessor(struct)
            graphs.append(graph_to_tensor_data(g) if isinstance(g, (nx.MultiDiGraph, nx.DiGraph, nx.Graph)) else g)
            valid_idx.append(i)
        except Exception as e:
            print(f"Error {i}: {e}")
    return graphs, valid_idx

print("Converting graphs...")
train_graphs, train_valid = convert_structures(train_structures)
train_targets = train_targets[train_valid]
val_graphs, val_valid = convert_structures(val_structures)
val_targets = val_targets[val_valid]
test_graphs, test_valid = convert_structures(test_structures)
test_targets = test_targets[test_valid]

config = model_default_nested.copy()
config['inputs'] = {k: v for k, v in config['inputs'].items() if k in ['atomic_number', 'multiplicity', 'edge_indices', 'offset', 'line_graph_edge_indices']}

def make_input_dict(graph_list):
    atomic = [g['atomic_number'] for g in graph_list]
    multi = [g['multiplicity'] for g in graph_list]
    e_idx = [g['edge_indices'] for g in graph_list]
    lg_idx = [g['line_graph_edge_indices'] for g in graph_list]
    off = [g['offset'] for g in graph_list]
    x_e_idx = tf.ragged.constant(e_idx, dtype=tf.int64, inner_shape=(2,), ragged_rank=1)
    x_lg_idx = tf.ragged.constant(lg_idx, dtype=tf.int64, inner_shape=(2,), ragged_rank=1)
    return {
        'atomic_number': tf.ragged.constant(atomic, dtype=tf.int32, ragged_rank=1),
        'multiplicity': tf.ragged.constant(multi, dtype=tf.int32, ragged_rank=1),
        'edge_indices': tf.cast(x_e_idx, dtype=tf.int32),
        'line_graph_edge_indices': tf.cast(x_lg_idx, dtype=tf.int32),
        'offset': tf.ragged.constant(off, dtype=tf.float32, ragged_rank=1)
    }

scaler = StandardScaler()
y_train = train_targets.reshape(-1, 1)
y_val = val_targets.reshape(-1, 1)
y_test = test_targets.reshape(-1, 1)
scaler.fit(y_train)

X_train = make_input_dict(train_graphs)
X_val = make_input_dict(val_graphs)
X_test = make_input_dict(test_graphs)

model = make_model(**config)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mae', metrics=['mae'])

print("Training...")
start = time.time()
model.fit(X_train, scaler.transform(y_train), validation_data=(X_val, scaler.transform(y_val)), epochs=50, batch_size=32, verbose=1)
train_time = time.time() - start

preds = scaler.inverse_transform(model.predict(X_test))
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print(f"\n{'='*70}")
print(f"RESULTS - coNGN on {SPLIT_TYPE}")
print(f"Test MAE: {mae:.4f}, Test RMSE: {rmse:.4f}")
print(f"{'='*70}")

results = {'model': 'coNGN', 'split_type': SPLIT_TYPE, 'test_mae': float(mae), 'test_rmse': float(rmse),
           'train_samples': len(train_graphs), 'val_samples': len(val_graphs), 'test_samples': len(test_graphs), 'train_time': train_time}
with open(RESULTS_DIR / f"coNGN_{SPLIT_TYPE}_results.json", 'w') as f:
    json.dump(results, f, indent=2)

pd.DataFrame({'true': y_test.flatten(), 'pred': preds.flatten()}).to_csv(RESULTS_DIR / f"coNGN_{SPLIT_TYPE}_predictions.csv", index=False)
model.save(RESULTS_DIR / f"coNGN_{SPLIT_TYPE}_model", save_format='tf')
print("SUCCESS!")
SCRIPT_END

    # Replace placeholder with actual split type
    sed -i "s/SPLIT_PLACEHOLDER/${split}/g" scripts/temp_coNGN_${split}.py

    # Create SLURM script
    cat > scripts/slurm_coNGN_${split}.slurm << EOF
#!/bin/bash
#SBATCH --job-name=coNGN_${split}
#SBATCH --output=/projects/d2r2/ardavan/ml4msd_group_project/logs/coNGN_${split}_%j.out
#SBATCH --error=/projects/d2r2/ardavan/ml4msd_group_project/logs/coNGN_${split}_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

source /shared/centos7/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate /projects/d2r2/ardavan/conda_envs/cogn_env
export PYTHONNOUSERSITE=1
cd /projects/d2r2/ardavan/ml4msd_group_project
python scripts/temp_coNGN_${split}.py
EOF

    # Submit job
    sbatch scripts/slurm_coNGN_${split}.slurm
    sleep 2
done

echo ""
echo "============================================"
echo "All coNGN jobs submitted!"
echo "Monitor with: squeue -u \$USER"
echo "============================================"
echo ""
echo "Once complete, collect results with:"
echo "for f in results/coNGN_matfold/coNGN_*_results.json; do echo \"--- \$(basename \$f) ---\"; cat \"\$f\"; done"