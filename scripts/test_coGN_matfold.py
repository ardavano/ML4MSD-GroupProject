"""
coGN with MatFold Splits - Uses pre-made OOD splits from Yue
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

# KGCNN imports
from kgcnn.literature.coGN import make_model, model_default
from kgcnn.crystal.preprocessor import KNNAsymmetricUnitCell
from kgcnn.data.transform.scaler.standard import StandardScaler

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

# ==========================================================================
# CONFIGURATION - Change these for different splits
# ==========================================================================
SPLIT_TYPE = "chemsys"  # Options: chemsys, composition, elements, periodictablegroups, pointgroup, sgnum
SPLIT_RATIO = "0.7-0.2-0.1"  # 70% train, 20% val, 10% test

SPLITS_DIR = Path("/projects/d2r2/ardavan/ml4msd_group_project/data/Splits-Yue")
RESULTS_DIR = Path("/projects/d2r2/ardavan/ml4msd_group_project/results/coGN_matfold")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Build file paths
base_name = f"mf.index-val_{SPLIT_TYPE}-test_{SPLIT_RATIO}"
train_file = SPLITS_DIR / f"{base_name}.train.csv"
val_file = SPLITS_DIR / f"{base_name}.validation.csv"
test_file = SPLITS_DIR / f"{base_name}.test.csv"

print("="*70)
print(f"coGN with MatFold Split: {SPLIT_TYPE}")
print(f"Split ratio: {SPLIT_RATIO}")
print("="*70)

# --------------------------------------------------------------------------
# Step 1: Load MatFold Split Data
# --------------------------------------------------------------------------
print("\nStep 1: Loading MatFold splits...")

def load_split_file(filepath):
    """Load a MatFold split CSV and recover pymatgen structures."""
    df = pd.read_csv(filepath)
    
    # Recover pymatgen Structure objects from dict strings
    structures = []
    for struct_str in df['structure']:
        struct_dict = eval(struct_str)  # Convert string to dict
        struct = Structure.from_dict(struct_dict)
        structures.append(struct)
    
    # Get target column (last column that's not mat_index or structure)
    target_col = [c for c in df.columns if c not in ['mat_index', 'structure']][0]
    targets = df[target_col].values
    
    return structures, targets, target_col

print(f"Loading train: {train_file.name}")
train_structures, train_targets, target_col = load_split_file(train_file)

print(f"Loading validation: {val_file.name}")
val_structures, val_targets, _ = load_split_file(val_file)

print(f"Loading test: {test_file.name}")
test_structures, test_targets, _ = load_split_file(test_file)

print(f"\nTarget property: {target_col}")
print(f"Train: {len(train_structures)}, Val: {len(val_structures)}, Test: {len(test_structures)}")

# --------------------------------------------------------------------------
# Step 2: Graph Conversion
# --------------------------------------------------------------------------
print("\nStep 2: Converting structures to graphs...")
start = time.time()

preprocessor = KNNAsymmetricUnitCell(k=12)

def graph_to_tensor_data(nx_graph):
    """Convert NetworkX graph to tensor-ready dict."""
    nodes_z = []
    nodes_mult = []
    
    for n in range(len(nx_graph.nodes)):
        node_data = nx_graph.nodes[n]
        nodes_z.append(node_data.get('atomic_number', 0))
        nodes_mult.append(node_data.get('multiplicity', 1))

    edges_u = []
    edges_v = []
    edges_offset = []
    
    for u, v, k, data in nx_graph.edges(keys=True, data=True):
        edges_u.append(u)
        edges_v.append(v)
        edges_offset.append(data.get('offset', [0, 0, 0]))
            
    return {
        'atomic_number': np.array(nodes_z, dtype="int32"),
        'multiplicity': np.array(nodes_mult, dtype="int32"),
        'edge_indices': np.array([edges_u, edges_v], dtype="int64").T,
        'offset': np.array(edges_offset, dtype="float32")
    }

def convert_structures_to_graphs(structures):
    """Convert list of pymatgen structures to graph data."""
    graphs = []
    valid_indices = []
    
    for i, struct in enumerate(structures):
        try:
            g = preprocessor(struct)
            if isinstance(g, (nx.MultiDiGraph, nx.DiGraph, nx.Graph)):
                g_data = graph_to_tensor_data(g)
            else:
                g_data = g 
            graphs.append(g_data)
            valid_indices.append(i)
        except Exception as e:
            print(f"  Warning: Error converting structure {i}: {e}")
    
    return graphs, valid_indices

# Convert all splits
print("  Converting training structures...")
train_graphs, train_valid = convert_structures_to_graphs(train_structures)
train_targets = train_targets[train_valid]

print("  Converting validation structures...")
val_graphs, val_valid = convert_structures_to_graphs(val_structures)
val_targets = val_targets[val_valid]

print("  Converting test structures...")
test_graphs, test_valid = convert_structures_to_graphs(test_structures)
test_targets = test_targets[test_valid]

prep_time = time.time() - start
print(f"Graph conversion complete: {prep_time:.1f}s")
print(f"Final counts - Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")

# --------------------------------------------------------------------------
# Step 3: Prepare Model Inputs
# --------------------------------------------------------------------------
print("\nStep 3: Preparing model inputs...")

config = model_default.copy()
inputs_to_keep = ['atomic_number', 'multiplicity', 'edge_indices', 'offset']
config['inputs'] = {k: v for k, v in config['inputs'].items() if k in inputs_to_keep}

def make_input_dict(graph_list):
    """Convert list of graph dicts to TensorFlow ragged tensors."""
    atomic_nums = [g['atomic_number'] for g in graph_list]
    multiplicities = [g['multiplicity'] for g in graph_list]
    edge_indices = [g['edge_indices'] for g in graph_list]
    offsets = [g['offset'] for g in graph_list]
    
    x_atomic = tf.ragged.constant(atomic_nums, dtype=tf.int32, ragged_rank=1)
    x_multi = tf.ragged.constant(multiplicities, dtype=tf.int32, ragged_rank=1)
    x_offset = tf.ragged.constant(offsets, dtype=tf.float32, ragged_rank=1)
    x_edge_ind = tf.ragged.constant(edge_indices, dtype=tf.int64, inner_shape=(2,), ragged_rank=1)
    x_edge_ind = tf.cast(x_edge_ind, dtype=tf.int32)

    return {
        'atomic_number': x_atomic,
        'multiplicity': x_multi,
        'edge_indices': x_edge_ind,
        'offset': x_offset
    }

# Scale targets
scaler = StandardScaler()
y_train = train_targets.reshape(-1, 1)
y_val = val_targets.reshape(-1, 1)
y_test = test_targets.reshape(-1, 1)

scaler.fit(y_train)
y_train_scaled = scaler.transform(y_train)
y_val_scaled = scaler.transform(y_val)

# Create input dicts
X_train = make_input_dict(train_graphs)
X_val = make_input_dict(val_graphs)
X_test = make_input_dict(test_graphs)

# --------------------------------------------------------------------------
# Step 4: Build and Train Model
# --------------------------------------------------------------------------
print("\nStep 4: Building model...")
model = make_model(**config)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mae', metrics=['mae'])

print("\nStep 5: Training (50 epochs)...")
start = time.time()
history = model.fit(
    X_train, y_train_scaled,
    validation_data=(X_val, y_val_scaled), 
    epochs=50, batch_size=32, verbose=1
)
train_time = time.time() - start

# --------------------------------------------------------------------------
# Step 6: Evaluate and Save
# --------------------------------------------------------------------------
print("\nStep 6: Evaluating...")
preds = scaler.inverse_transform(model.predict(X_test))
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print(f"\n{'='*70}")
print(f"RESULTS - coGN on {SPLIT_TYPE} split")
print(f"{'='*70}")
print(f"Test MAE:  {mae:.4f}")
print(f"Test RMSE: {rmse:.4f}")
print(f"Train time: {train_time:.1f}s")
print(f"{'='*70}")

# Save results
results = {
    'model': 'coGN',
    'split_type': SPLIT_TYPE,
    'split_ratio': SPLIT_RATIO,
    'test_mae': float(mae),
    'test_rmse': float(rmse),
    'train_samples': len(train_graphs),
    'val_samples': len(val_graphs),
    'test_samples': len(test_graphs),
    'epochs': 50,
    'train_time_seconds': train_time
}

result_file = RESULTS_DIR / f"coGN_{SPLIT_TYPE}_{SPLIT_RATIO}_results.json"
with open(result_file, 'w') as f:
    json.dump(results, f, indent=2)

# Save predictions
pred_df = pd.DataFrame({
    'true': y_test.flatten(),
    'predicted': preds.flatten()
})
pred_file = RESULTS_DIR / f"coGN_{SPLIT_TYPE}_{SPLIT_RATIO}_predictions.csv"
pred_df.to_csv(pred_file, index=False)

# Save model
model_dir = RESULTS_DIR / f"coGN_{SPLIT_TYPE}_{SPLIT_RATIO}_model"
model.save(model_dir, save_format='tf')

print(f"\nResults saved to: {result_file}")
print(f"Predictions saved to: {pred_file}")
print(f"Model saved to: {model_dir}")
print("\nSUCCESS!")