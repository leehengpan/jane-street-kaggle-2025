import torch
import torch.optim as optim
import wandb
import time
import itertools
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.custom_dataset import CustomDataset
from src.nn import MLP
from src.attention_nn import ComplexMLPWithAttention

import polars as pl

import datetime
import os





def calc_r2(y_preds: torch.Tensor, y_vals: torch.Tensor, weights: torch.Tensor, is_loss=False) -> float:
    sum_square_residuals = torch.sum(weights * (y_vals - y_preds) ** 2)
    total_sum_squares = torch.sum(weights * (y_vals ** 2)) + 1e-6
    r2 = 1 - (sum_square_residuals / total_sum_squares)
    if is_loss:
        return -r2
    else:
        return r2.item()


def evaluate(model, val_dataloader, device):
    y_preds = [] # save predicted output
    y_vals = [] # save the ground truth
    weights = [] # save weights

    early_stop = 100000 * 10
    count = 0
    
    model.eval()
    with torch.no_grad():
        for x_val, c_val, y_val in val_dataloader:
            x_val, c_val, y_val = x_val.to(device), c_val.to(device), y_val.to(device)
            x_weight = x_val[:, 0]
            y_pred = model(x_val, c_val)

            y_preds.append(y_pred)
            y_vals.append(y_val)
            weights.append(x_weight)

            count += x_val.shape[0]
            if count >= early_stop:
                break
    
    y_preds = torch.cat(y_preds)
    y_vals = torch.cat(y_vals)
    weights = torch.cat(weights)

    r2 = calc_r2(y_preds, y_vals, weights)
    return r2

def training(data_path, 
             val_data_path,
             cat_mapping_paths, 
             stats_path, 
             batch_size=32, 
             lr=1e-3, 
             dropout=0.4, 
             hidden_layers=[64, 64, 128], 
             emb_dims=16, 
             proj_dims=64, 
             total_iters=10000, 
             log_every=20, 
             eval_every=200,
             save_every=1000,
             r2_loss=False,
             val100k_only=False,
             mask_p=0.0,
             alpha=0.7
             ):

    outdir = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_lr{lr}_log" # Time-based unique directory
    if os.path.exists(outdir):
        print(f"The directory {outdir} already exists.")
    else:
        os.makedirs(outdir)
        print(f"Created unique out_dir... {outdir}")

    
    print("Initializing wandb...")
    wandb.init(project='mlp_training', config={
        'learning_rate': lr,
        'dropout': dropout,
        'hidden_layers': hidden_layers,
        'embedding_dims': emb_dims,
        'projection_dims': proj_dims,
        'batch_size': batch_size,
        'total_iterations': total_iters,
        'mask_p': mask_p,
        'r2_loss': r2_loss
    })

    print("Initializing datasets...")
    train_dataset = CustomDataset(data_path, cat_mapping_paths, stats_path)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_dataloader = itertools.cycle(train_dataloader)

    total_k = len(train_dataset)/1000
    print(f"len(train_dataset): {total_k}k")

    # Validation set (if validation data is provided)
    if val_data_path:
        val_dataset = CustomDataset(val_data_path, cat_mapping_paths, stats_path)
        val_dataloader = DataLoader(val_dataset, batch_size=100000, shuffle=False)

        val_total_k = len(val_dataset)/1000
        print(f"len(val_dataset): {val_total_k}k, val100k_only:{val100k_only}")

    else:
        val_dataloader = None

    print("Initializing model...")
    feat_in_dims = len(train_dataset.cont_features)
    cat_in_dims = [len(train_dataset.cat_mapping[f"feature_{i:02d}"])+1 for i in range(9, 12)] # +1 for unseen category
    model = MLP(feat_in_dims, cat_in_dims, hidden_layers, emb_dims, proj_dims, dropout)

    # model = ComplexMLPWithAttention(feat_in_dims, cat_in_dims, hidden_layers, emb_dims, proj_dims, dropout)
    # hidden_layers = [128, 256, 512, 256, 128] 
    # emb_dims = 32 
    # proj_dims = 128 
    # dropout = 0.1/0.3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Set up the optimizer (Adam) and loss function (L2 loss)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    l2_loss = torch.nn.MSELoss() 
    l1_loss = torch.nn.L1Loss()

    start_time = time.time()

    print("Start training...")
    for step in range(1, total_iters + 1):

        model.train()
        
        x, c, y = next(train_dataloader) 
        x, c, y = x.to(device), c.to(device), y.to(device)

        # Random mask_p of c -> 0
        mask = torch.rand(c.shape, device=c.device) >= mask_p 
        c = c * mask

        optimizer.zero_grad()
        output = model(x, c)
        if r2_loss:
            weights = x[:, 0]
            loss = calc_r2(output, y, weights, is_loss=True)
        else:
            loss = alpha * l1_loss(output, y) + (1-alpha) * l2_loss(output, y)
        loss.backward()
        optimizer.step()

        if step % log_every == 0:
            training_dur = time.time() - start_time  # Time spent on training step
            wandb.log({f"train_loss, r2_loss: {r2_loss}": loss.item()}, step=step)
            seen_k = (batch_size * step) / 1000
            percent= (seen_k / total_k) * 100
            print(f"Step {step}/{total_iters}, Training Loss: {loss.item()}, r2_loss: {r2_loss}")
            print(f"Step {step}/{total_iters}, Total: {total_k:.2f}K, Seen: {seen_k:.2f}K ({percent:.2f}%)")
            print(f"Step {step}/{total_iters}, Time: {training_dur:.2f} seconds")

        if val_dataloader and step % eval_every == 0:
            start_val_time = time.time()
            model.eval()

            if r2_loss:
                val_loss = evaluate(model, val_dataloader, device)
            else:
                val_loss = 0
                accum_samples = 0
                batch_count = 0
                with torch.no_grad():
                    for x_val, c_val, y_val in val_dataloader:
                        x_val, c_val, y_val = x_val.to(device), c_val.to(device), y_val.to(device)
                        val_output = model(x_val, c_val)
                        val_loss += (alpha * l1_loss(val_output, y_val) + (1-alpha) * l2_loss(val_output, y_val)).item()
                        accum_samples += batch_size
                        batch_count += 1
                val_loss /= batch_count

            val_dur = time.time() - start_val_time  
            wandb.log({"val_loss": val_loss}, step=step)
            print(f"Step {step}/{total_iters}, Validation Loss: {val_loss}, r2_loss: {r2_loss}")
            print(f"Step {step}/{total_iters}, Val Time: {val_dur:.2f} seconds")

        if step % save_every == 0:
            model_save_path = os.path.join(outdir, f"mlp_model_step_{step}.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at {model_save_path}")

    # Final model save after training is complete
    torch.save(model.state_dict(), "mlp_model_final.pth")
    print("Training Complete and Final Model Saved!")


