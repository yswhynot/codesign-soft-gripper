import os
import re
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from train import MLPModel

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MLP model on test data")
    parser.add_argument("--test_data_file", type=str, required=True,
                        help="Path to the test NPZ data file")
    parser.add_argument("--object_name", type=str, required=True,
                        help="Name of the object to evaluate")
    parser.add_argument("--density", type=float, default=2e0,
                        help="Density of the object")
    return parser.parse_args()

def parse_model_filename(model_paths, layer_num=None, dropout=None, hidden_dim=None, batch_size=None, lr=1e-3, density=1e0, pointnet_dim=128, data_amount=None):
    """
    Given a list of model file paths, this function parses each filename to extract architecture
    parameters and then filters the list based on any provided parameters.
    
    Expected filename format example:
      006_mustard_bottle_frame550_layer4_drop0.2_hidden512_batch256_lr0.0005_weight0.001_l1_model.pth
    (and optionally includes something like 'density100.0')
    
    Args:
      model_paths (list of str): List of model file paths.
      layer_num (int, optional): If provided, filter for files with this layer number.
      dropout (float, optional): If provided, filter for files with this dropout value.
      hidden_dim (int, optional): If provided, filter for files with this hidden dimension.
      density (float, optional): If provided, filter for files with this density value.
    
    Returns:
      list of tuples: Each tuple is (model_path, parsed_layer_num, parsed_dropout, parsed_hidden_dim, parsed_density)
                      for files that match all provided filters.
    """
    for model_path in model_paths:
        basename = os.path.basename(model_path)
        basename = os.path.splitext(basename)[0]  # remove extension
        
        try:
            parsed_layer = int(re.search(r'layer(\d+)', basename).group(1))
            parsed_dropout = float(re.search(r'drop(\d*\.?\d+)', basename).group(1))
            parsed_hidden = int(re.search(r'hidden(\d+)', basename).group(1))
            parsed_batch = int(re.search(r'batch(\d+)', basename).group(1))
            parsed_lr = float(re.search(r'lr(\d*\.?\d+)', basename).group(1))
            parsed_density = 1.0
            
        except Exception as e:
            # If any parameter is missing or fails to parse, skip this file
            continue
        
        if layer_num is not None and parsed_layer != layer_num:
            continue
        if dropout is not None and parsed_dropout != dropout:
            continue
        if hidden_dim is not None and parsed_hidden != hidden_dim:
            continue
        if batch_size is not None and parsed_batch != batch_size:
            continue
        if lr is not None and parsed_lr != lr:
            continue
        
        return parsed_layer, parsed_dropout, parsed_hidden, parsed_density, pointnet_dim, model_path
    raise FileNotFoundError("No matching model file found.")

def load_object_model(model_dir, object_name, name=""):
    if name:
        return [os.path.join(model_dir, name)]
    pattern = os.path.join(model_dir, f"{object_name}*")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No file starting with '{object_name}' found in '{model_dir}'.")
    # Optionally, you could sort the list if you need a consistent choice:
    matches.sort()
    return matches

def evaluate(model, dataloader, criterion, device, collect_samples=False):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_ground_truth = []
    
    with torch.no_grad():
        
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)
            running_loss += loss.item() * x_batch.size(0)
            
            if collect_samples:
                all_preds.append(predictions.cpu())
                all_ground_truth.append(y_batch.cpu())
                
    avg_loss = running_loss / len(dataloader.dataset)
    
    if collect_samples:
        all_preds = torch.cat(all_preds, dim=0)
        all_ground_truth = torch.cat(all_ground_truth, dim=0)
        return avg_loss, all_preds, all_ground_truth
    else:
        return avg_loss

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    curr_dir = os.path.dirname(os.path.abspath(__file__))

    # Load test data from NPZ file
    test_data_file = os.path.join(curr_dir, args.test_data_file)
    test_data = np.load(test_data_file, allow_pickle=True)
    X_test_np = test_data["log_k"]   # shape (N, input_dim)
    Y_test_np = test_data["output"]  # shape (N, output_dim)

    X_test = torch.from_numpy(X_test_np).float().to(device)
    Y_test = torch.from_numpy(Y_test_np).float().to(device)
    
    # Load the saved model checkpoint (which contains normalization parameters)
    model_paths = load_object_model(curr_dir + '/../sim_models/', args.object_name)
    
    # Parse model architecture parameters from the model filename
    layer_num, dropout, hidden_dim, density, model_path = parse_model_filename(model_paths, layer_num=4, dropout=0.2, hidden_dim=1024, batch_size=512, lr=1e-3, density=args.density)
    print("Parsed parameters from filename:")
    print(f"  layer_num: {layer_num}")
    print(f"  dropout: {dropout}")
    print(f"  hidden_dim: {hidden_dim}")
    print(f"  density: {density}")

    checkpoint = torch.load(model_path, map_location=device)
    model_state = checkpoint["model_state_dict"]
    norm_params = checkpoint["norm_params"]

    # Get normalization parameters and ensure they are on the proper device
    X_mean = norm_params["X_mean"].to(device)
    X_std  = norm_params["X_std"].to(device)
    Y_mean = norm_params["Y_mean"].to(device)
    Y_std  = norm_params["Y_std"].to(device)

    # Normalize test data using the training statistics
    X_test_norm = (X_test - X_mean) / X_std
    Y_test_norm = (Y_test - Y_mean) / Y_std

    test_dataset = TensorDataset(X_test_norm, Y_test_norm)
    test_loader = DataLoader(test_dataset, shuffle=False)


    input_dim = X_test_norm.shape[1]
    output_dim = Y_test_norm.shape[1]
    # Recreate the model using parameters from the filename
    model = MLPModel(input_dim, output_dim, hidden_dim=hidden_dim,
                     layer_num=layer_num, dropout=dropout)
    model.load_state_dict(model_state)
    model.to(device)

    # Choose loss function
    criterion = nn.L1Loss()

    # Evaluate model on test data
    # Evaluate and collect samples
    test_loss, preds, gt = evaluate(model, test_loader, criterion, device, collect_samples=True)
    print(f"Test loss: {test_loss:.4f}")
    print("Sample un-normalized predictions vs. ground truth:")
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    for i in range(min(10, preds.size(0))):
        # Get the i-th normalized prediction and ground truth
        norm_pred = preds[i]       # shape: [output_dim]
        norm_gt   = gt[i]
        
        # Denormalize: elementwise operation on the whole vector
        denorm_pred = norm_pred * norm_params["Y_std"].cpu() + norm_params["Y_mean"].cpu()
        denorm_gt   = norm_gt   * norm_params["Y_std"].cpu() + norm_params["Y_mean"].cpu()
        
        print(f"Sample {i}:")
        print(f"  Unnormalized Prediction: {denorm_pred.numpy()}")
        print(f"  Unnormalized Ground Truth: {denorm_gt.numpy()}")
        print(f"  Total force ground truth: {np.linalg.norm(denorm_gt.numpy()[0:6])}")
        print(f"  Total q ground truth: {np.linalg.norm(denorm_gt.numpy()[6:9])}")
        print(f"  Total force prediction: {np.linalg.norm(denorm_pred.numpy()[0:6])}")
        print(f"  Total q prediction: {np.linalg.norm(denorm_pred.numpy()[6:9])}")

if __name__ == "__main__":
    main()