import polars as pl
import numpy as np
import gc
import os

# create a class to generate lag statistics 
# responder_i_lag_1_min, responder_i_lag_1_max, responder_i_lag_1_mean, responder_i_lag_i_sum for responder_0 to responder_8

class CreateLagStats:
    def __init__(self, lag_file, out_dir):
        self.lag_file = lag_file
        self.out_dir = out_dir
        self.responder_cols = [f"responder_{i}_lag_1" for i in range(9)]
        self.df = pl.scan_parquet(self.lag_file).collect()
        self.lag_stats = self._calculate_lag_stats()
    
    def _calculate_lag_stats(self):
        print("Calculating lag stats...")
        
        # Create aggregation expressions for each responder
        agg_exprs = []
        for col in self.responder_cols:
            agg_exprs.append(pl.col(col).min().alias(f"{col}_min"))
            agg_exprs.append(pl.col(col).max().alias(f"{col}_max"))
            agg_exprs.append(pl.col(col).mean().alias(f"{col}_mean"))
            agg_exprs.append(pl.col(col).sum().alias(f"{col}_sum"))
        
        # Calculate lag statistics grouped by date_id
        lag_stats = self.df.group_by("date_id").agg(agg_exprs)
        
        print(f"Lag stats shape: {lag_stats.shape}")
        return lag_stats

    def _save_lag_stats(self):
        print("Saving lag stats...")
        self.lag_stats.write_parquet(os.path.join(self.out_dir, "lag_stats.parquet"))

if __name__ == "__main__":
    lag_file = "./dataset/train.parquet"
    out_dir = "./lag_stats"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    lag_stats = CreateLagStats(lag_file, out_dir)
    lag_stats._save_lag_stats()