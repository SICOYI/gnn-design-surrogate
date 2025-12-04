import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader, HeteroData, Batch
from torch_geometric.nn import GCNConv, HeteroConv, SAGEConv, Linear, to_hetero
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
import io
from collections import defaultdict
warnings.filterwarnings('ignore')

BATCH_SIZE = 16
EPOCHS = 1000
LEARNING_RATE = 0.1
HIDDEN_DIM = 64
FORCE_INPUT_DELIMITER = None

def get_script_directory():
    return os.path.dirname(os.path.abspath(__file__))

def find_all_csv_tsv_files(directory):
    file_paths = []
    if not os.path.exists(directory):
        return []

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.csv', '.tsv')):
            file_paths.append(file_path)
    return file_paths

def determine_delimiter(file_path, first_line_content):
    if FORCE_INPUT_DELIMITER:
        return FORCE_INPUT_DELIMITER

    num_commas = first_line_content.count(',')
    num_tabs = first_line_content.count('\t')

    if num_tabs > num_commas and num_tabs > 0:
        return '\t'
    elif num_commas > num_tabs and num_commas > 0:
        return ','
    elif file_path.lower().endswith('.tsv'):
        return '\t'
    elif file_path.lower().endswith('.csv'):
        return ','
    
    return ','

def classify_structural_elements(conn_df, coord_df):
    node_coords = {}
    for _, row in coord_df.iterrows():
        node_id = row['NodeID']
        node_coords[node_id] = {
            'X': float(row['X']),
            'Y': float(row['Y']),
            'Z': float(row['Z'])
        }
    
    element_labels = []
    
    for _, row in conn_df.iterrows():
        start_node = row['StartNodeID']
        end_node = row['EndNodeID']
        
        start_coords = node_coords.get(start_node)
        end_coords = node_coords.get(end_node)
        
        if start_coords is None or end_coords is None:
            element_labels.append(-1)
            continue
        
        dx = abs(start_coords['X'] - end_coords['X'])
        dy = abs(start_coords['Y'] - end_coords['Y'])
        dz = abs(start_coords['Z'] - end_coords['Z'])
        
        if dz > 0 and dx == 0 and dy == 0:
            label = 1
        elif dz == 0 and (dx > 0 or dy > 0):
            label = 0
        else:
            total_length = np.sqrt(dx**2 + dy**2 + dz**2)
            if total_length > 0 and dz / total_length > 0.7:
                label = 1
            else:
                label = 0
        
        element_labels.append(label)
    
    return np.array(element_labels)

def load_and_process_hetero_data():
    script_dir = get_script_directory()
    data_source_dir = script_dir
    
    data_file_paths = find_all_csv_tsv_files(data_source_dir)
    
    if not data_file_paths:
        print("No data files found.")
        return None
    
    print(f"Files: {len(data_file_paths)}")
    
    all_conn_data = {}
    all_coord_data = {}
    all_frame_def_data = {}
    
    for file_path in data_file_paths:
        file_name = os.path.basename(file_path)
        
        try:
            with open(file_path, 'r', newline='', encoding='utf-8') as f:
                raw_content = f.read()
            
            if not raw_content.strip():
                continue
            
            first_line = raw_content.splitlines()[0]
            delimiter = determine_delimiter(file_path, first_line)
            
            df = pd.read_csv(io.StringIO(raw_content), delimiter=delimiter)
            df.columns = df.columns.str.strip()
            
            if 'Dataset_ID' not in df.columns:
                continue
            
            is_connections = all(col in df.columns for col in ['ElemID', 'StartNodeID', 'EndNodeID'])
            is_coordinates = all(col in df.columns for col in ['NodeID', 'X', 'Y', 'Z'])
            is_frame_def = all(col in df.columns for col in ['family', 'height_cm', 'Flange_width_cm', 'Area_cm2'])
            
            for dataset_id, group_df in df.groupby('Dataset_ID'):
                key = (file_name, dataset_id)
                
                if is_connections:
                    all_conn_data[key] = group_df
                elif is_coordinates:
                    all_coord_data[key] = group_df
                elif is_frame_def:
                    for col in ['height_cm', 'Flange_width_cm', 'Area_cm2']:
                        if col in group_df.columns:
                            group_df[col] = pd.to_numeric(group_df[col], errors='coerce')
                    all_frame_def_data[key] = group_df
        
        except Exception as e:
            continue
    
    if not all_conn_data or not all_coord_data or not all_frame_def_data:
        return None
    
    print("Building heterogeneous graphs...")
    
    conn_by_dataset = defaultdict(list)
    coord_by_dataset = defaultdict(list)
    frame_by_dataset = defaultdict(list)
    
    for (filename, dataset_id), df in all_conn_data.items():
        conn_by_dataset[dataset_id].append(df)
    
    for (filename, dataset_id), df in all_coord_data.items():
        coord_by_dataset[dataset_id].append(df)
    
    for (filename, dataset_id), df in all_frame_def_data.items():
        frame_by_dataset[dataset_id].append(df)
    
    common_dataset_ids = set(conn_by_dataset.keys()) & set(coord_by_dataset.keys()) & set(frame_by_dataset.keys())
    common_dataset_ids = sorted(list(common_dataset_ids))
    
    print(f"Datasets: {len(common_dataset_ids)}")
    
    hetero_graphs = {}
    
    for dataset_id in common_dataset_ids:
        try:
            conn_df = pd.concat(conn_by_dataset[dataset_id], ignore_index=True)
            coord_df = pd.concat(coord_by_dataset[dataset_id], ignore_index=True)
            frame_df = pd.concat(frame_by_dataset[dataset_id], ignore_index=True)
            
            conn_df = conn_df.sort_values('ElemID').reset_index(drop=True)
            
            element_labels = classify_structural_elements(conn_df, coord_df)
            
            all_node_ids = set()
            all_node_ids.update(conn_df['StartNodeID'].unique())
            all_node_ids.update(conn_df['EndNodeID'].unique())
            all_node_ids.update(coord_df['NodeID'].unique())
            
            node_id_to_idx = {node_id: idx for idx, node_id in enumerate(sorted(all_node_ids))}
            num_nodes = len(node_id_to_idx)
            
            node_features = torch.zeros((num_nodes, 3), dtype=torch.float)
            for _, row in coord_df.iterrows():
                node_id = row['NodeID']
                if node_id in node_id_to_idx:
                    idx = node_id_to_idx[node_id]
                    node_features[idx, 0] = float(row['X'])
                    node_features[idx, 1] = float(row['Y'])
                    node_features[idx, 2] = float(row['Z'])
            
            frame_df = frame_df.reset_index(drop=True)
            
            if len(conn_df) != len(frame_df):
                min_len = min(len(conn_df), len(frame_df))
                conn_df = conn_df.iloc[:min_len]
                frame_df = frame_df.iloc[:min_len]
                element_labels = element_labels[:min_len]
            
            beam_edge_index = []
            column_edge_index = []
            beam_features = []
            column_features = []
            
            for idx, row in conn_df.iterrows():
                src = node_id_to_idx[row['StartNodeID']]
                dst = node_id_to_idx[row['EndNodeID']]
                
                height = float(frame_df.iloc[idx]['height_cm']) if not pd.isna(frame_df.iloc[idx]['height_cm']) else 0.0
                flange = float(frame_df.iloc[idx]['Flange_width_cm']) if not pd.isna(frame_df.iloc[idx]['Flange_width_cm']) else 0.0
                
                if idx < len(element_labels):
                    if element_labels[idx] == 0:
                        beam_edge_index.append([src, dst])
                        beam_features.append([height, flange])
                    elif element_labels[idx] == 1:
                        column_edge_index.append([src, dst])
                        column_features.append([height, flange])
            
            hetero_data = HeteroData()
            
            hetero_data['node'].x = node_features
            
            if len(beam_edge_index) > 0:
                beam_edge_index = torch.tensor(beam_edge_index, dtype=torch.long).t().contiguous()
                hetero_data['node', 'beam', 'node'].edge_index = beam_edge_index
                hetero_data['node', 'beam', 'node'].y = torch.tensor(beam_features, dtype=torch.float)
            
            if len(column_edge_index) > 0:
                column_edge_index = torch.tensor(column_edge_index, dtype=torch.long).t().contiguous()
                hetero_data['node', 'column', 'node'].edge_index = column_edge_index
                hetero_data['node', 'column', 'node'].y = torch.tensor(column_features, dtype=torch.float)
            
            hetero_data.dataset_id = dataset_id
            hetero_data.num_nodes = num_nodes
            
            hetero_graphs[dataset_id] = hetero_data
            
            print(f"Dataset {dataset_id}: {len(beam_edge_index[0]) if len(beam_edge_index) > 0 else 0} beams, "
                  f"{len(column_edge_index[0]) if len(column_edge_index) > 0 else 0} columns")
            
        except Exception as e:
            print(f"Error building graph for dataset {dataset_id}: {e}")
            continue
    
    print(f"Heterogeneous graphs built: {len(hetero_graphs)}")
    return hetero_graphs

class HeteroGNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv = HeteroConv({
            ('node', 'beam', 'node'): GCNConv(in_channels, out_channels),
            ('node', 'column', 'node'): GCNConv(in_channels, out_channels),
        }, aggr='sum')
    
    def forward(self, x_dict, edge_index_dict):
        return self.conv(x_dict, edge_index_dict)

class HeterogeneousGNN(nn.Module):
    def __init__(self, node_dim=3, edge_dim=2, hidden_dim=64):
        super().__init__()
        
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        self.gnn1 = HeteroGNNLayer(hidden_dim, hidden_dim)
        self.gnn2 = HeteroGNNLayer(hidden_dim, hidden_dim)
        
        self.beam_predictor = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, edge_dim)
        )
        
        self.column_predictor = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, edge_dim)
        )
        
        self.edge_type_predictor = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)
        )
    
    def forward(self, data):
        x_dict = {'node': self.node_encoder(data['node'].x)}
        
        edge_index_dict = {}
        if ('node', 'beam', 'node') in data.edge_types:
            edge_index_dict[('node', 'beam', 'node')] = data['node', 'beam', 'node'].edge_index
        if ('node', 'column', 'node') in data.edge_types:
            edge_index_dict[('node', 'column', 'node')] = data['node', 'column', 'node'].edge_index
        
        x_dict = self.gnn1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = self.gnn2(x_dict, edge_index_dict)
        
        node_features = x_dict['node']
        
        beam_preds, column_preds = None, None
        edge_type_preds = None
        
        all_edge_predictions = []
        all_edge_types = []
        
        if ('node', 'beam', 'node') in data.edge_types:
            beam_edges = data['node', 'beam', 'node'].edge_index
            beam_src, beam_dst = beam_edges[0], beam_edges[1]
            beam_features = torch.cat([node_features[beam_src], node_features[beam_dst]], dim=1)
            beam_preds = self.beam_predictor(beam_features)
            all_edge_predictions.append(beam_preds)
            all_edge_types.extend([0] * len(beam_preds))
            
            if self.training:
                edge_type_preds_beam = self.edge_type_predictor(beam_features)
                if edge_type_preds is None:
                    edge_type_preds = edge_type_preds_beam
                else:
                    edge_type_preds = torch.cat([edge_type_preds, edge_type_preds_beam], dim=0)
        
        if ('node', 'column', 'node') in data.edge_types:
            column_edges = data['node', 'column', 'node'].edge_index
            column_src, column_dst = column_edges[0], column_edges[1]
            column_features = torch.cat([node_features[column_src], node_features[column_dst]], dim=1)
            column_preds = self.column_predictor(column_features)
            all_edge_predictions.append(column_preds)
            all_edge_types.extend([1] * len(column_preds))
            
            if self.training:
                edge_type_preds_column = self.edge_type_predictor(column_features)
                if edge_type_preds is None:
                    edge_type_preds = edge_type_preds_column
                else:
                    edge_type_preds = torch.cat([edge_type_preds, edge_type_preds_column], dim=0)
        
        return beam_preds, column_preds, edge_type_preds, torch.tensor(all_edge_types, device=node_features.device)

def hetero_collate_fn(batch):
    batched_data = Batch.from_data_list(batch)
    return batched_data

def train_hetero_model(model, train_loader, val_loader, epochs=500, lr=0.01):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.9
    )
    
    train_losses = []
    val_losses = []
    
    patience = 20
    min_delta = 1e-3
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("Training heterogeneous GNN...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_batches = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            beam_pred, column_pred, edge_type_pred, true_edge_types = model(batch)
            
            total_loss = 0
            main_loss = 0
            
            if beam_pred is not None and hasattr(batch['node', 'beam', 'node'], 'y'):
                beam_loss = F.mse_loss(beam_pred, batch['node', 'beam', 'node'].y)
                total_loss += beam_loss
                main_loss += beam_loss.item()
            
            if column_pred is not None and hasattr(batch['node', 'column', 'node'], 'y'):
                column_loss = F.mse_loss(column_pred, batch['node', 'column', 'node'].y)
                total_loss += column_loss
                main_loss += column_loss.item()
            
            if edge_type_pred is not None:
                aux_loss = F.cross_entropy(edge_type_pred, true_edge_types)
                total_loss += 0.1 * aux_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += main_loss
            train_batches += 1
        
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                beam_pred, column_pred, _, _ = model(batch)
                
                batch_loss = 0
                if beam_pred is not None and hasattr(batch['node', 'beam', 'node'], 'y'):
                    batch_loss += F.mse_loss(beam_pred, batch['node', 'beam', 'node'].y).item()
                if column_pred is not None and hasattr(batch['node', 'column', 'node'], 'y'):
                    batch_loss += F.mse_loss(column_pred, batch['node', 'column', 'node'].y).item()
                
                val_loss += batch_loss
                val_batches += 1
        
        train_loss_avg = train_loss / train_batches if train_batches > 0 else 0
        val_loss_avg = val_loss / val_batches if val_batches > 0 else 0
        
        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)
        
        scheduler.step(val_loss_avg)
        
        if epoch >= 10:
            recent_val_losses = val_losses[-10:]
            avg_recent_loss = np.mean(recent_val_losses)
            improvement = (avg_recent_loss - val_loss_avg) / avg_recent_loss if avg_recent_loss > 0 else 0
            
            if improvement < min_delta:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                patience_counter = 0
                best_val_loss = min(best_val_loss, val_loss_avg)
        
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
            print(f'Epoch {epoch+1:03d}/{epochs} | '
                  f'Train MSE: {train_loss_avg:.1f} | '
                  f'Val MSE: {val_loss_avg:.1f} | '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
    return train_losses, val_losses

def plot_losses(train_losses, val_losses, save_path):
    epochs_range = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(epochs_range, train_losses, 'b-', label='Train MSE Loss', linewidth=2)
    axes[0, 0].plot(epochs_range, val_losses, 'r-', label='Val MSE Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE Loss')
    axes[0, 0].set_title('Training and Validation MSE Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(epochs_range, train_losses, 'b-', label='Train MSE', linewidth=2)
    axes[0, 1].plot(epochs_range, val_losses, 'r-', label='Val MSE', linewidth=2)
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MSE Loss (log scale)')
    axes[0, 1].set_title('MSE Loss (Log Scale)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    final_train_loss = train_losses[-1] if train_losses else 0
    final_val_loss = val_losses[-1] if val_losses else 0
    best_val_loss = min(val_losses) if val_losses else 0
    
    metrics_text = f"Training Summary:\n"
    metrics_text += f"Final Train MSE: {final_train_loss:.2f}\n"
    metrics_text += f"Final Val MSE: {final_val_loss:.2f}\n"
    metrics_text += f"Best Val MSE: {best_val_loss:.2f}\n"
    metrics_text += f"Total Epochs: {len(train_losses)}\n"
    metrics_text += f"Improvement: {train_losses[0]-final_train_loss:.1f} (Train)\n"
    metrics_text += f"Improvement: {val_losses[0]-final_val_loss:.1f} (Val)"
    
    axes[1, 0].text(0.1, 0.5, metrics_text, fontsize=12, 
                   verticalalignment='center', horizontalalignment='left',
                   transform=axes[1, 0].transAxes,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 0].axis('off')
    axes[1, 0].set_title('Training Summary')
    
    if len(train_losses) > 20:
        last_50 = min(50, len(train_losses))
        axes[1, 1].plot(range(last_50), train_losses[-last_50:], 'b-', label='Train (last 50)', linewidth=2)
        axes[1, 1].plot(range(last_50), val_losses[-last_50:], 'r-', label='Val (last 50)', linewidth=2)
        axes[1, 1].set_xlabel('Epoch (Recent)')
    else:
        axes[1, 1].plot(epochs_range, train_losses, 'b-', label='Train', linewidth=2)
        axes[1, 1].plot(epochs_range, val_losses, 'r-', label='Val', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
    
    axes[1, 1].set_ylabel('MSE Loss')
    axes[1, 1].set_title('Recent Loss Trends')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_filename = os.path.join(save_path, 'hetero_training_loss_plot.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Heterogeneous GNN loss plot saved to: {plot_filename}")
    plt.show()

def main():
    print("True Heterogeneous GNN Training for Edge Feature Prediction")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    script_dir = get_script_directory()
    print(f"Script dir: {script_dir}")
    
    print("\n1. Loading and building heterogeneous data...")
    hetero_graphs = load_and_process_hetero_data()
    
    if hetero_graphs is None or len(hetero_graphs) == 0:
        print("No data found.")
        return
    
    print("\n2. Preparing heterogeneous data...")
    dataset = list(hetero_graphs.values())
    
    train_indices, val_indices = train_test_split(
        range(len(dataset)), 
        test_size=0.2, 
        random_state=42
    )
    
    train_dataset = [dataset[i] for i in train_indices]
    val_dataset = [dataset[i] for i in val_indices]
    
    print(f"Train graphs: {len(train_dataset)}, Val graphs: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=hetero_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=hetero_collate_fn)
    
    print("\n3. Creating true heterogeneous GNN model...")
    model = HeterogeneousGNN(node_dim=3, edge_dim=2, hidden_dim=HIDDEN_DIM)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    print("\n4. Training heterogeneous model...")
    train_losses, val_losses = train_hetero_model(
        model, train_loader, val_loader, 
        epochs=EPOCHS, lr=LEARNING_RATE
    )
    
    model_save_path = os.path.join(script_dir, 'true_heterogeneous_gnn_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'config': {
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'learning_rate': LEARNING_RATE,
            'hidden_dim': HIDDEN_DIM,
            'actual_epochs_trained': len(train_losses),
            'model_type': 'TrueHeterogeneousGNN'
        }
    }, model_save_path)
    
    print(f"\nModel saved to: {model_save_path}")
    print(f"Total epochs trained: {len(train_losses)}")
    
    plot_losses(train_losses, val_losses, script_dir)
    
    print("\n5. Validation results:")
    model.eval()
    
    for i, graph in enumerate(val_dataset[:2]):
        print(f"\nGraph {i+1} (ID: {graph.dataset_id}):")
        graph = graph.to(device)
        
        with torch.no_grad():
            beam_pred, column_pred, _, _ = model(graph)
        
        total_mae = 0
        total_edges = 0
        
        if beam_pred is not None and hasattr(graph['node', 'beam', 'node'], 'y'):
            beam_mae = torch.mean(abs(beam_pred - graph['node', 'beam', 'node'].y)).item()
            print(f"Beams: {len(beam_pred)}, MAE: {beam_mae:.1f}")
            total_mae += beam_mae * len(beam_pred)
            total_edges += len(beam_pred)
            
            if len(beam_pred) > 0:
                print("First 2 beam predictions:")
                for idx in range(min(2, len(beam_pred))):
                    true = graph['node', 'beam', 'node'].y[idx]
                    pred = beam_pred[idx]
                    print(f"  Beam {idx}: True({true[0]:.0f},{true[1]:.0f}) Pred({pred[0]:.0f},{pred[1]:.0f})")
        
        if column_pred is not None and hasattr(graph['node', 'column', 'node'], 'y'):
            column_mae = torch.mean(abs(column_pred - graph['node', 'column', 'node'].y)).item()
            print(f"Columns: {len(column_pred)}, MAE: {column_mae:.1f}")
            total_mae += column_mae * len(column_pred)
            total_edges += len(column_pred)
            
            if len(column_pred) > 0:
                print("First 2 column predictions:")
                for idx in range(min(2, len(column_pred))):
                    true = graph['node', 'column', 'node'].y[idx]
                    pred = column_pred[idx]
                    print(f"  Column {idx}: True({true[0]:.0f},{true[1]:.0f}) Pred({pred[0]:.0f},{pred[1]:.0f})")
        
        if total_edges > 0:
            print(f"Overall weighted MAE: {total_mae/total_edges:.1f}")

if __name__ == "__main__":
    try:
        import torch_geometric
        main()
    except ImportError:
        print("Please install required packages: pip install torch torch-geometric pandas numpy scikit-learn matplotlib")