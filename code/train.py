import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import random

def parse_args():
    parser = argparse.ArgumentParser(description="Train MLP model for stiffness-to-output mapping")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10000, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=1024, help="Initial hidden dimension")
    parser.add_argument("--layer_num", type=int, default=5, help="Number of hidden layers")
    parser.add_argument("--pointnet_dim", type=int, default=128, help="PointNet layer dimension")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--loss_type", type=str, default="l1", help="Loss function type")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="Weight decay for optimizer")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--object_name", type=str, default="all", help="Name of the object to sample")
    parser.add_argument("--scheduler_patience", type=int, default=200, help="Patience for the LR scheduler (epochs)")
    parser.add_argument("--early_stop_patience", type=int, default=500, help="Early stopping patience (epochs)")
    parser.add_argument("--lr_factor", type=float, default=0.8, help="Factor by which to reduce LR on plateau")
    parser.add_argument("--log_prefix", type=str, default="", help="Prefix for TensorBoard logs")
    parser.add_argument("--train_option", type=int, default=1, help="1. all data; 2. 1 pose all stiffness; 3. 1 stiffness all poses")
    return parser.parse_args()

class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3, output_dim=128):
        super(PointNetEncoder, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, output_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(output_dim)
        self.global_feat = global_feat
        self.output_dim = output_dim

    def forward(self, x, mode='train'):
        B, D, N = x.size()

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.output_dim)
        return x

class MLPModel(nn.Module):
    def __init__(self, nopt_dim, output_dim, hidden_dim=128, layer_num=4, dropout=0.2, pointnet_dim=256):
        super().__init__()
        self.pn_encoder = PointNetEncoder(channel=3, output_dim=pointnet_dim)
        self.pc_scale = nn.Parameter(torch.ones(1))
        layers = []
        # First layer: input -> hidden_dim
        layers.append(nn.Linear(nopt_dim + pointnet_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout))
        
        # Add (layer_num - 1) hidden layers with the same dimension
        # (Subtract one because the first layer is already added)
        for _ in range(layer_num - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))

        # Penultimate layer: reduce dimension from hidden_dim to hidden_dim//2
        layers.append(nn.Linear(hidden_dim, hidden_dim // 2))
        layers.append(nn.BatchNorm1d(hidden_dim // 2))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout))
        
        # Final output layer: map from hidden_dim//2 to output_dim
        layers.append(nn.Linear(hidden_dim // 2, output_dim))
        
        self.net = nn.Sequential(*layers)

    def forward(self, pc, x, mode='train'):
        """
        pc: [B, 3, N]  (point cloud for each object)
        stiffness: [B, 22]
        """
        # pc_feat = self.pn_encoder(pc, mode=mode) 
        pc_feat = self.pc_scale * self.pn_encoder(pc, mode=mode)
        # combined = torch.cat([pc_feat, stiffness], dim=1)
        combined = torch.cat([pc_feat, x], dim=1)
        out = self.net(combined) 
        return out

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for pc_batch, x_batch, y_batch in dataloader:
            pc_batch = pc_batch.to(device)
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            predictions = model(pc_batch, x_batch, mode='eval')  # Combined model expects (point cloud, stiffness)
            loss = criterion(predictions, y_batch)
            running_loss += loss.item() * pc_batch.size(0)
    return running_loss / len(dataloader.dataset)

def load_data(curr_dir, object_name, train_option=1):
    frame = 850
    num_points = 1024
    data_dir = curr_dir + f'/../data_gen/{object_name}_frame{frame}/rand/'
    com_dir_file = curr_dir + '/../models/coms/coms.npz'
    pc_dir = curr_dir + f'/../models/partial_pointclouds/{object_name}'

    # search for a file in the pc_dir for the object_name_num_points
    pc_names = []
    pc_data = []
    repeated_pc = []
    
    for file in os.listdir(pc_dir):
        pc_file = os.path.join(pc_dir, file)
        # make sure pc_file ends with .npz
        if not pc_file.endswith(".npz"):
            continue
        pc_names.append(file.split(f'sample{num_points}')[1][1:])
        pc_data.append(np.load(pc_file, allow_pickle=True)['pc'].T)
        
    # search in the data_dir for all files 
    data = {'log_k': [], 'density': [], 'output': []}
    # max_len = 40
    if train_option == 1:
        # max_len = 40
        for i, file in enumerate(pc_names):
            data_file = os.path.join(data_dir, file)
            this_data = np.load(data_file, allow_pickle=True)
        
            N = this_data['log_k'].shape[0]
            if ('log_k' in this_data) and ('output' in this_data)and ('density_list' in this_data):
                data['log_k'].append(this_data['log_k'])
                output = this_data['output']
                # make collision only 0 or 1
                output[:, -1] = (output[:, -1] > 0.0).astype(float)
                data['output'].append(output)
                data['density'].append(this_data['density_list'])
                pc = pc_data[i]
                r_pc = np.repeat(pc[np.newaxis, :, :], N, axis=0)
                repeated_pc.append(r_pc)
            else:
                print(f"Warning: {data_file} does not contain required keys 'log_k' or 'output'. Skipping.")
                
    elif train_option == 2:
        selected_indices = random.sample(range(len(pc_names)), 10)
        for i in selected_indices:
            file = pc_names[i]
            data_file = os.path.join(data_dir, file)
            this_data = np.load(data_file, allow_pickle=True)
            N = this_data['log_k'].shape[0]
            if 'log_k' in this_data and 'output' in this_data:
                data['log_k'].append(this_data['log_k'][:N, :])
                data['output'].append(this_data['output'][:N, :])
                pc = pc_data[i]
                r_pc = np.repeat(pc[np.newaxis, :, :], N, axis=0)
                repeated_pc.append(r_pc)
            else:
                print(f"Warning: {data_file} does not contain required keys 'log_k' or 'output'. Skipping.")
        
    elif train_option == 3:
        for i, file in enumerate(pc_names):
            data_file = os.path.join(data_dir, file)
            this_data = np.load(data_file, allow_pickle=True)
            num = 11
            indices = random.sample(range(this_data['log_k'].shape[0]), num)
            if 'log_k' in this_data and 'output' in this_data:
                data['log_k'].append(this_data['log_k'][indices, :])
                data['output'].append(this_data['output'][indices, :])
                pc = pc_data[i]
                r_pc = np.repeat(pc[np.newaxis, :, :], num, axis=0)
                repeated_pc.append(r_pc)
            else:
                print(f"Warning: {data_file} does not contain required keys 'log_k' or 'output'. Skipping.")

    data['log_k'] = np.vstack(data['log_k']) # shape (N, 22)
    data['output'] = np.vstack(data['output']) # shape (N, 9)
    data['density'] = np.vstack(data['density']) # shape (N, 1)
    repeated_pc = np.vstack(repeated_pc) # shape (N, 3, num_points)
    N = data["log_k"].shape[0]
    com_info = np.load(com_dir_file, allow_pickle=True)['com_dict'].item()[object_name]
    
    data['com'] = np.repeat(com_info[np.newaxis, :], N, axis=0)
    
    return data, repeated_pc

def normalize_point_cloud(pc_tensor, 
                          centroids=None, max_distance=None):
    """
    Normalize each point cloud sample (assumed shape [B, 3, N]) independently:
      - Subtract the centroid
      - Scale so that the maximum distance from the centroid is 1.
    
    Returns:
      pc_normalized: Normalized point cloud, same shape as input.
      centroids: Centroid of each sample, shape [B, 3, 1].
      scales: Max distance (scale factor) for each sample, shape [B, 1, 1].
    """
    B, D, N = pc_tensor.shape
    if centroids is None:
        # Compute centroid along the points dimension for each sample.
        centroids = pc_tensor.mean(dim=2, keepdim=True)  # shape: [B, 3, 1]
    pc_tensor -= centroids
    
    if max_distance is None:
        distances = torch.norm(pc_tensor, dim=1)  
        max_distance = distances.max(dim=1, keepdim=True)[0].unsqueeze(2)
        # breakpoint()
        max_distance[max_distance < 1e-6] = 1e-6  # avoid division by zero
    
    # Scale the centered point cloud so that max distance is 1.
    pc_normalized = pc_tensor / max_distance
    return pc_normalized, centroids, max_distance

def get_object_names(data_dir):
    """Get all object names from the data directory."""
    object_names = []
    for file in os.listdir(data_dir):
        name = file.split('_frame')[0]  # Extract name before '_frame'
        if name not in object_names:
            object_names.append(name)
    return object_names

def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Determine current directory
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    object_list = get_object_names(curr_dir + '/../data_gen/')
    print("object_list:", object_list)
    print("Loading data...")
    
    X_np, Y_np, pc_data = [], [], []
    
    for i, obj_name in enumerate(object_list):        
        data, pc = load_data(curr_dir, obj_name, train_option=args.train_option)
        # X_np = data["log_k"]   # shape (N, input_dim)
        X_np.append(np.hstack([data["log_k"], data["com"], data["density"]]))  # shape (N, 22 + 3 + 1)
        Y_np.append(data["output"])  # shape (N, output_dim)
        pc_data.append(pc)  # shape (3, N)
    
    X_np = np.vstack(X_np)  # shape (total_N, 22 + 3 + 1)
    Y_np = np.vstack(Y_np)  # shape (total_N, output_dim)
    pc_data = np.vstack(pc_data)  # shape (total_N, 3, num_points)
    print("X_np shape:", X_np.shape)
    print("Y_np shape:", Y_np.shape)
    print("pc_data shape:", pc_data.shape)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert to torch tensors (without normalization yet) 
    X_tensor = torch.from_numpy(X_np).float().to(device)
    Y_tensor = torch.from_numpy(Y_np).float().to(device)
    pc_tensor = torch.from_numpy(pc_data).float().to(device)

    N = X_tensor.shape[0]  

    print("Normalizing point cloud...")
    pc_norm, pc_centroids, pc_scales = normalize_point_cloud(pc_tensor)
    
    X_tensor[:, 22:25] = normalize_point_cloud(
                            X_tensor[:, 22:25].unsqueeze(2),
                            centroids=pc_centroids,
                            max_distance=pc_scales)[0].squeeze(2)
    
    print("Normalized pc_tensor mean:", pc_norm.mean(), "std:", pc_norm.std())
    
    # Create full dataset: each sample is (point_cloud, stiffness, output)
    full_dataset = TensorDataset(pc_norm, X_tensor, Y_tensor)
    N = len(full_dataset)
    
    # Split into train and validation sets
    val_size = int(args.val_split * N)
    train_size = N - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Compute normalization stats for stiffness and output from training set only
    train_X = torch.cat([train_dataset[i][1].unsqueeze(0) for i in range(len(train_dataset))], dim=0)
    train_Y = torch.cat([train_dataset[i][2].unsqueeze(0) for i in range(len(train_dataset))], dim=0)
    X_mean = train_X.mean(dim=0)
    X_std = train_X.std(dim=0)
    X_std[X_std < 1e-6] = 1e-6
    Y_mean = train_Y.mean(dim=0)
    Y_std = train_Y.std(dim=0)
    Y_std[Y_std < 1e-6] = 1e-6
    
    # Normalization function for stiffness and output
    def normalize_dataset(dataset, X_mean, X_std, Y_mean, Y_std):
        normalized = []
        for pc, x, y in dataset:
            x_norm = (x - X_mean) / X_std
            x_norm[22:] = x[22:]
            y_norm = (y - Y_mean) / Y_std
            normalized.append((pc, x_norm, y_norm))
            # normalized.append((pc, x, y_norm))
        return normalized
    
    train_dataset_norm = normalize_dataset(train_dataset, X_mean, X_std, Y_mean, Y_std)
    val_dataset_norm = normalize_dataset(val_dataset, X_mean, X_std, Y_mean, Y_std)
    
    # Build DataLoaders
    train_loader = DataLoader(train_dataset_norm, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset_norm, batch_size=args.batch_size, shuffle=False)
    
    # ---------------------------------------------------
    # 5. Initialize Model, Optimizer, Loss, and TensorBoard
    # ---------------------------------------------------
    input_dim = X_tensor.shape[1]   # 22 + 3 + 1
    output_dim = Y_tensor.shape[1]  # 10
    model = MLPModel(nopt_dim=input_dim, output_dim=output_dim,
                          hidden_dim=args.hidden_dim, layer_num=args.layer_num, dropout=args.dropout, pointnet_dim=args.pointnet_dim)
    print("Input dim:", input_dim, "Output dim:", output_dim)
    
    model.to(device)
    
    optimizer = optim.AdamW([
        {'params': model.pn_encoder.parameters(), 
         'lr': args.lr * 0.5},
        {'params': model.net.parameters(), 
         'lr': args.lr}
    ], weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    if args.loss_type == "l1":
        criterion = nn.L1Loss()
    
    # Set up learning rate scheduler (ReduceLROnPlateau)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_factor,
                                  patience=args.scheduler_patience, verbose=True)
    if args.train_option == 1:
        file_name = f"{args.object_name}_layer{args.layer_num}_drop{args.dropout}_hidden{args.hidden_dim}_pointout{args.pointnet_dim}_batch{args.batch_size}_lr{args.lr}_weight{args.weight_decay}_seed{args.seed}_{args.loss_type}_{args.log_prefix}_alldata"
    elif args.train_option == 2:
        file_name = f"{args.object_name}_layer{args.layer_num}_drop{args.dropout}_hidden{args.hidden_dim}_pointout{args.pointnet_dim}_batch{args.batch_size}_lr{args.lr}_weight{args.weight_decay}_{args.loss_type}_{args.log_prefix}_1pose"
    elif args.train_option == 3:
        file_name = f"{args.object_name}_layer{args.layer_num}_drop{args.dropout}_hidden{args.hidden_dim}_pointout{args.pointnet_dim}_batch{args.batch_size}_lr{args.lr}_weight{args.weight_decay}_{args.loss_type}_{args.log_prefix}_allpose"

    log_dir = f"../runs/{file_name}"
    tb_log_dir = os.path.join(curr_dir, log_dir)
    writer = SummaryWriter(log_dir=tb_log_dir)
    
    best_val_loss = float("inf")
    best_model_path = os.path.join(curr_dir, "../sim_models/" + file_name + "_model.pth")
    epochs_since_improvement = 0
    
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.1
        for pc_batch, x_batch, y_batch in train_loader:
            
            pc_batch = pc_batch.to(device)
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            predictions = model(pc_batch, x_batch, mode='train')
            loss = criterion(predictions, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * pc_batch.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        writer.add_scalar("Loss/Train", train_loss, epoch)
    
        if epoch % 100 == 0:
            val_loss = evaluate(model, val_loader, criterion, device)

            writer.add_scalar("Loss/Val", val_loss, epoch)
            print(f"Epoch [{epoch+1}/{args.epochs}]: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
    
        scheduler.step(val_loss)
    
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            norm_params = {
                "X_mean": X_mean,
                "X_std": X_std,
                "Y_mean": Y_mean,
                "Y_std": Y_std
            }
            torch.save({"model_state_dict": model.state_dict(), "norm_params": norm_params}, best_model_path)
            epochs_since_improvement = 0
        elif val_loss >= best_val_loss * 2.0:
            epochs_since_improvement += 1
    
        if epochs_since_improvement >= args.early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    writer.close()
    print(f"Training complete. Best val_loss={best_val_loss:.4f}. Model saved at {best_model_path}")

if __name__ == "__main__":
    main()