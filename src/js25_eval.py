import torch
from torch.utils.data import DataLoader
from src.nn import MLP
from src.custom_dataset import CustomDataset
from tqdm import tqdm
import argparse
import json

def calc_r2(y_preds: torch.Tensor, y_vals: torch.Tensor, weights: torch.Tensor) -> float:
    sum_square_residuals = torch.sum(weights * (y_vals - y_preds) ** 2)
    total_sum_squares = torch.sum(weights * (y_vals ** 2))
    if total_sum_squares == 0:
        return 1.0 if sum_square_residuals == 0 else 0.0
    r2 = 1 - (sum_square_residuals / total_sum_squares)
    return r2.item()

def evaluate(model, val_dataloader, device):
    y_preds = []  
    y_vals = []   
    weights = []

    model.eval()
    with torch.no_grad():
        for x_val, c_val, y_val in tqdm(val_dataloader, desc="Evaluating", leave=True):
            x_val, c_val, y_val = x_val.to(device), c_val.to(device), y_val.to(device)
            x_weight = x_val[:, 0]  
            y_pred = model(x_val, c_val)
            y_pred = torch.clamp(y_pred, min=-5, max=5)
            y_preds.append(y_pred)
            y_vals.append(y_val)
            weights.append(x_weight)

    y_preds = torch.cat(y_preds)
    y_vals = torch.cat(y_vals)
    weights = torch.cat(weights)
    r2 = calc_r2(y_preds, y_vals, weights)
    return r2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate MLP model")
    parser.add_argument('--model-save-path', type=str, required=True, help="Path to the saved model")
    parser.add_argument('--val-data-path', type=str, required=True, help="Path to the validation dataset")
    parser.add_argument('--stats-path', type=str, required=True, help="Path to stats file")
    parser.add_argument('--feat-in-dims', type=int, required=True, help="Number of continuous input dimensions")
    parser.add_argument('--cat-in-dims', type=int, nargs='+', required=True, help="List of categorical input dimensions")
    parser.add_argument('--hidden-layers', type=int, nargs='+', required=True, help="List of hidden layer dimensions")
    parser.add_argument('--emb-dims', type=int, required=True, help="Embedding dimensions for categorical inputs")
    parser.add_argument('--proj-dims', type=int, required=True, help="Projection output dimensions")
    parser.add_argument('--dropout', type=float, required=True, help="Dropout rate")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cat_mapping_paths = {f"feature_{i:02d}": f"./preprocessed_dataset/feature_{i:02d}_cat_mapping.parquet" for i in range(9, 12)}

    model = MLP(args.feat_in_dims, args.cat_in_dims, args.hidden_layers, args.emb_dims, args.proj_dims, args.dropout).to(device)
    model.load_state_dict(torch.load(args.model_save_path, map_location=device))

    val_dataset = CustomDataset(args.val_data_path, cat_mapping_paths, args.stats_path)
    val_dataloader = DataLoader(val_dataset, batch_size=100000, shuffle=False)

    r2 = evaluate(model, val_dataloader, device)
    print(f"R2 score: {r2}")
