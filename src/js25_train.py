import torch
import argparse
from src.training import training

def parse_args():
    parser = argparse.ArgumentParser(description="Training Script for MLP Model")
    
    # Add arguments for each hyperparameter
    parser.add_argument('--data_path', type=str, default='./preprocessed_dataset/training.parquet', help="Path to training dataset")
    parser.add_argument('--val_data_path', type=str, default='./preprocessed_dataset/validation.parquet', help="Path to validation dataset")
    parser.add_argument('--cat_mapping_paths', type=str, default='{f"feature_{i:02d}": f"./preprocessed_dataset/feature_{i:02d}_cat_mapping.parquet" for i in range(9, 12)}', help="Paths for categorical feature mappings")
    parser.add_argument('--stats_path', type=str, default='./preprocessed_dataset/stats.parquet', help="Path to dataset statistics")
    parser.add_argument('--batch_size', type=int, default=1024, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--dropout', type=float, default=0.4, help="Dropout rate")
    parser.add_argument('--hidden_layers', type=int, nargs='+', default=[64, 64, 128], help="List of hidden layer sizes")
    parser.add_argument('--emb_dims', type=int, default=16, help="Embedding dimensions")
    parser.add_argument('--proj_dims', type=int, default=64, help="Projection dimensions")
    parser.add_argument('--total_iters', type=int, default=10000, help="Total number of iterations")
    parser.add_argument('--log_every', type=int, default=20, help="Log training loss every N steps")
    parser.add_argument('--eval_every', type=int, default=200, help="Evaluate the model every N steps")
    parser.add_argument('--save_every', type=int, default=2000, help="Save the model every N steps")
    parser.add_argument('--r2_loss', type=bool, default=False, help="Use R2 loss instead of MSE")
    parser.add_argument('--val100k_only', type=bool, default=False, help="Validate 100k random samples from validation set")
    parser.add_argument('--tf32', type=bool, default=False, help="Use tf32.")
    parser.add_argument('--mask_p', type=float, default=0.0, help="Mask categorical probability")
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    print(args)
    
    # Check if CUDA is available
    is_cuda_available = torch.cuda.is_available()
    print("Check GPU available:", is_cuda_available)
    assert is_cuda_available, "CUDA is not available!"

    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    print("Use tf32:", args.tf32)
    
    # Convert cat_mapping_paths from string to dictionary
    cat_mapping_paths = eval(args.cat_mapping_paths)

    # Call the training function
    training(
        data_path=args.data_path,
        val_data_path=args.val_data_path,
        cat_mapping_paths=cat_mapping_paths,
        stats_path=args.stats_path,
        batch_size=args.batch_size,
        lr=args.lr,
        dropout=args.dropout,
        hidden_layers=args.hidden_layers,
        emb_dims=args.emb_dims,
        proj_dims=args.proj_dims,
        total_iters=args.total_iters,
        log_every=args.log_every,
        eval_every=args.eval_every,
        save_every=args.save_every,
        r2_loss=args.r2_loss,
        val100k_only=args.val100k_only,
        mask_p=args.mask_p
    )

if __name__ == '__main__':
    main()
