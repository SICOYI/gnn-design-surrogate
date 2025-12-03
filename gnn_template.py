import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
import io
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

def load_and_process_data():
    print("Loading data...")
    
    script_dir = get_script_directory()
    data_source_dir = script_dir
    
    print(f"Data dir: {data_source_dir}")
    
    data_file_paths = find_all_csv_tsv_files(data_source_dir)
    
    if not data_file_paths:
        print("No data files found.")
        return None
    
    print(f"Files: {len(data_file_paths)}")
    
    all_conn_data = {}
    all_coord_data = {}
    all_frame_def_data = {}
    all_plate_def_data = {}
    
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
            is_plate_def = all(col in df.columns for col in ['family', 'face1_height_cm'])
            
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
                elif is_plate_def:
                    for col in ['face1_height_cm', 'face2_height_cm']:
                        if col in group_df.columns:
                            group_df[col] = pd.to_numeric(group_df[col], errors='coerce')
                    all_plate_def_data[key] = group_df
        
        except Exception as e:
            continue
    
    if not all_conn_data:
        print("No connection data")
        return None
    if not all_coord_data:
        print("No coordinate data")
        return None
    if not all_frame_def_data:
        print("No frame cross-section data")
        return None
    
    print("Building graphs...")
    
    conn_by_dataset = {}
    coord_by_dataset = {}
    frame_by_dataset = {}
    plate_by_dataset = {}
    
    for (filename, dataset_id), df in all_conn_data.items():
        conn_by_dataset.setdefault(dataset_id, []).append(df)
    
    for (filename, dataset_id), df in all_coord_data.items():
        coord_by_dataset.setdefault(dataset_id, []).append(df)
    
    for (filename, dataset_id), df in all_frame_def_data.items():
        frame_by_dataset.setdefault(dataset_id, []).append(df)
    
    for (filename, dataset_id), df in all_plate_def_data.items():
        plate_by_dataset.setdefault(dataset_id, []).append(df)
    
    common_dataset_ids = set(conn_by_dataset.keys())
    common_dataset_ids = common_dataset_ids.intersection(coord_by_dataset.keys())
    common_dataset_ids = common_dataset_ids.intersection(frame_by_dataset.keys())
    common_dataset_ids = sorted(list(common_dataset_ids))
    
    print(f"Datasets: {len(common_dataset_ids)}")
    
    graphs_data = {}
    
    for dataset_id in common_dataset_ids:
        try:
            conn_df = pd.concat(conn_by_dataset[dataset_id], ignore_index=True)
            coord_df = pd.concat(coord_by_dataset[dataset_id], ignore_index=True)
            frame_df = pd.concat(frame_by_dataset[dataset_id], ignore_index=True)
            
            conn_df = conn_df.sort_values('ElemID').reset_index(drop=True)
            
            all_node_ids = set()
            all_node_ids.update(conn_df['StartNodeID'].unique())
            all_node_ids.update(conn_df['EndNodeID'].unique())
            all_node_ids.update(coord_df['NodeID'].unique())
            
            node_id_to_idx = {node_id: idx for idx, node_id in enumerate(sorted(all_node_ids))}
            
            edge_index = []
            for _, row in conn_df.iterrows():
                src = node_id_to_idx[row['StartNodeID']]
                dst = node_id_to_idx[row['EndNodeID']]
                edge_index.append([src, dst])
            
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            
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
                edge_index = edge_index[:, :min_len]
                conn_df = conn_df.iloc[:min_len]
                frame_df = frame_df.iloc[:min_len]
            
            edge_features = []
            for _, row in frame_df.iterrows():
                height = float(row['height_cm']) if not pd.isna(row['height_cm']) else 0.0
                flange = float(row['Flange_width_cm']) if not pd.isna(row['Flange_width_cm']) else 0.0
                edge_features.append([height, flange])
            
            edge_features = torch.tensor(edge_features, dtype=torch.float)
            
            graph_data = Data(
                x=node_features,
                edge_index=edge_index,
                y=edge_features,
                dataset_id=dataset_id,
                num_nodes=num_nodes
            )
            
            graphs_data[dataset_id] = graph_data
            
        except Exception as e:
            continue
    
    print(f"Graphs built: {len(graphs_data)}")
    return graphs_data

class EnhancedEdgeGNN(nn.Module):
    def __init__(self, node_dim=3, edge_dim=2, hidden_dim=64):
        super(EnhancedEdgeGNN, self).__init__()
        
        self.conv1 = GCNConv(node_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        self.edge_predictor = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, edge_dim)
        )
        
    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        x = self.conv3(x, data.edge_index)
        
        src, dst = data.edge_index[0], data.edge_index[1]
        edge_features = torch.cat([x[src], x[dst]], dim=1)
        
        edge_pred = self.edge_predictor(edge_features)
        return edge_pred

def train_model(model, train_loader, val_loader, epochs=500, lr=0.01):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.9
    )
    
    train_losses = []
    val_losses = []
    mae_histories = {'height': [], 'flange': []}
    
    patience = 20
    min_delta = 1e-3
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop = False
    
    print("Training...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_batches = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            pred = model(batch)
            loss = F.mse_loss(pred, batch.y)
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        model.eval()
        val_loss = 0
        val_mae_height = 0
        val_mae_flange = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch)
                
                loss = F.mse_loss(pred, batch.y)
                mae_height = F.l1_loss(pred[:, 0], batch.y[:, 0]).item()
                mae_flange = F.l1_loss(pred[:, 1], batch.y[:, 1]).item()
                
                val_loss += loss.item()
                val_mae_height += mae_height
                val_mae_flange += mae_flange
                val_batches += 1
        
        train_loss_avg = train_loss / train_batches
        val_loss_avg = val_loss / val_batches
        val_mae_height_avg = val_mae_height / val_batches
        val_mae_flange_avg = val_mae_flange / val_batches
        
        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)
        mae_histories['height'].append(val_mae_height_avg)
        mae_histories['flange'].append(val_mae_flange_avg)
        
        scheduler.step(val_loss_avg)
        
        if epoch >= 10:
            recent_val_losses = val_losses[-10:]
            avg_recent_loss = np.mean(recent_val_losses)
            
            improvement = (avg_recent_loss - val_loss_avg) / avg_recent_loss
            
            if improvement < min_delta:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    early_stop = True
            else:
                patience_counter = 0
                best_val_loss = min(best_val_loss, val_loss_avg)
        
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1 or early_stop:
            print(f'Epoch {epoch+1:03d}/{epochs} | '
                  f'Loss: {train_loss_avg:.1f}/{val_loss_avg:.1f} | '
                  f'MAE H: {val_mae_height_avg:.1f} F: {val_mae_flange_avg:.1f} | '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        if early_stop:
            break
    
    return train_losses, val_losses, mae_histories

def plot_losses(train_losses, val_losses, mae_histories, save_path):
    epochs_range = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(epochs_range, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs_range, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(epochs_range, mae_histories['height'], 'g-', label='Height MAE', linewidth=2)
    axes[0, 1].plot(epochs_range, mae_histories['flange'], 'orange', label='Flange MAE', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE (cm)')
    axes[0, 1].set_title('Height and Flange MAE')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    final_train_loss = train_losses[-1] if train_losses else 0
    final_val_loss = val_losses[-1] if val_losses else 0
    final_height_mae = mae_histories['height'][-1] if mae_histories['height'] else 0
    final_flange_mae = mae_histories['flange'][-1] if mae_histories['flange'] else 0
    
    metrics_text = f"Final Metrics:\n"
    metrics_text += f"Train Loss: {final_train_loss:.2f}\n"
    metrics_text += f"Val Loss: {final_val_loss:.2f}\n"
    metrics_text += f"Height MAE: {final_height_mae:.1f} cm\n"
    metrics_text += f"Flange MAE: {final_flange_mae:.1f} cm\n"
    metrics_text += f"Total Epochs: {len(train_losses)}"
    
    axes[1, 0].text(0.1, 0.5, metrics_text, fontsize=12, 
                   verticalalignment='center', horizontalalignment='left',
                   transform=axes[1, 0].transAxes,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 0].axis('off')
    axes[1, 0].set_title('Final Metrics')
    
    if len(train_losses) > 20:
        last_50 = min(50, len(train_losses))
        axes[1, 1].plot(range(last_50), train_losses[-last_50:], 'b-', label='Last 50 Train', linewidth=2)
        axes[1, 1].plot(range(last_50), val_losses[-last_50:], 'r-', label='Last 50 Val', linewidth=2)
    else:
        axes[1, 1].plot(epochs_range, train_losses, 'b-', label='Train', linewidth=2)
        axes[1, 1].plot(epochs_range, val_losses, 'r-', label='Val', linewidth=2)
    
    axes[1, 1].set_xlabel('Epoch (Recent)')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Recent Loss Trends')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_filename = os.path.join(save_path, 'training_loss_plot.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Loss plot saved to: {plot_filename}")
    plt.show()

def main():
    print("GNN Training")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    script_dir = get_script_directory()
    print(f"Script dir: {script_dir}")
    
    print("\n1. Loading data...")
    graphs_data = load_and_process_data()
    
    if graphs_data is None or len(graphs_data) == 0:
        print("No data found.")
        return
    
    print("\n2. Preparing data...")
    dataset = list(graphs_data.values())
    
    train_indices, val_indices = train_test_split(
        range(len(dataset)), 
        test_size=0.2, 
        random_state=42
    )
    
    train_dataset = [dataset[i] for i in train_indices]
    val_dataset = [dataset[i] for i in val_indices]
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print("\n3. Creating model...")
    model = EnhancedEdgeGNN(node_dim=3, edge_dim=2, hidden_dim=HIDDEN_DIM)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {total_params:,}")
    
    print("\n4. Training...")
    train_losses, val_losses, mae_histories = train_model(
        model, train_loader, val_loader, 
        epochs=EPOCHS, lr=LEARNING_RATE
    )
    
    model_save_path = os.path.join(script_dir, 'gnn_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'mae_histories': mae_histories,
        'config': {
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'learning_rate': LEARNING_RATE,
            'hidden_dim': HIDDEN_DIM,
            'actual_epochs_trained': len(train_losses)
        }
    }, model_save_path)
    
    print(f"\nModel saved to: {model_save_path}")
    print(f"Total epochs trained: {len(train_losses)}")
    
    plot_losses(train_losses, val_losses, mae_histories, script_dir)
    
    print("\n5. Validation results:")
    model.eval()
    
    for i, graph in enumerate(val_dataset[:3]):
        print(f"\nGraph {i+1} (ID: {graph.dataset_id}):")
        graph = graph.to(device)
        
        with torch.no_grad():
            predictions = model(graph)
        
        mse = F.mse_loss(predictions, graph.y).item()
        mae_height = F.l1_loss(predictions[:, 0], graph.y[:, 0]).item()
        mae_flange = F.l1_loss(predictions[:, 1], graph.y[:, 1]).item()
        
        print(f"Nodes: {graph.num_nodes}, Edges: {graph.edge_index.shape[1]}")
        print(f"MSE: {mse:.1f}, Height MAE: {mae_height:.1f}, Flange MAE: {mae_flange:.1f}")
        
        print("Predictions (first 3):")
        for edge_idx in range(3):
            true_h = graph.y[edge_idx, 0].item()
            true_f = graph.y[edge_idx, 1].item()
            pred_h = predictions[edge_idx, 0].item()
            pred_f = predictions[edge_idx, 1].item()
            print(f"Edge {edge_idx}: T({true_h:.0f},{true_f:.0f}) P({pred_h:.0f},{pred_f:.0f})")

if __name__ == "__main__":
    try:
        import torch_geometric
        main()
    except ImportError:
        print("Install: pip install torch torch-geometric pandas numpy scikit-learn")