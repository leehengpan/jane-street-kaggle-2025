import pandas as pd
import polars as pl
import numpy as np
import os



# TODO: Data preprocess and save
# 1. Fill nan values, if nan values found in the beginning, fill with median, else fill with ffill.
# 2. Extend responder_i_lag from the previous day to the same row. 
#   - Last time_id of the previous day. 
# 3. Save all data to a parquet file. 
# 4. Compute mean of features, then save it as an independent dataframe called mean.parquet.
# 5. Map categorical values (id=9,10,11) to integer values, then save it as an independent dataframe called map_categorical.parquet.

class DataPreprocess:
    def __init__(self, train_file, train_start_date, train_end_date, val_start_date, val_end_date, out_dir):
        print("Initializing DataPreprocess...")
        self.train_file = train_file
        self.train_start_date = train_start_date
        self.train_end_date = train_end_date
        self.val_start_date = val_start_date
        self.val_end_date = val_end_date
        self.out_dir = out_dir
        self.categorical_columns = ["feature_09", "feature_10", "feature_11"]
        self.feature_with_beginning_nan_blocks = ["feature_00", "feature_01", "feature_02", "feature_03", "feature_04", "feature_15", "feature_16", "feature_17", "feature_18", "feature_19","feature_21", "feature_26", "feature_27", 
                                                  "feature_31", "feature_32", "feature_33", "feature_39", "feature_40", "feature_41", "feature_42", "feature_43", "feature_44", "feature_45", "feature_46", "feature_47", "feature_50",
                                                  "feature_52","feature_53","feature_55", "feature_56", "feature_57", "feature_58", "feature_62", "feature_63", "feature_64", "feature_65", "feature_66", "feature_73", "feature_74"]
        self.lag_cols_original = ["date_id", "symbol_id"] + [f"responder_{idx}" for idx in range(9)]
        self.lag_cols_rename = { f"responder_{idx}" : f"responder_{idx}_lag_1" for idx in range(9)}
        self.df = pl.scan_parquet(self.train_file).collect() 

        print("Finding all columns...")
        self.all_columns = [col for col in self.df.columns]
        print("Finding columns with NaN values...")
        self.nan_all_columns = [col for col in self.all_columns if self.df.select(pl.col(col).is_null().sum().alias('null_count'))[0,0] > 0]
        
        self._fill_nan_values()
        self._add_lags_stats()

    def _fill_nan_values(self):
        """
        Fill NaN values: If NaNs are found at the beginning, fill with median; otherwise, use forward fill.
        """
        print("Filling NaN values...")
        self.df = self.df.with_columns([
            pl.col("symbol_id").cum_count().over("symbol_id").alias("row_id")
        ])
            
        # features might have non-consecutive NaNs in the middle
        # here we only deal with the consecutive NaNs at the beginning
        for col in self.feature_with_beginning_nan_blocks:
            print(f"Filling consecutive NaNs in the beginning with median values for {col}...")

            # Compute the median values for each symbol_id group
            median_values = self.df.group_by("symbol_id").agg([
                pl.col(col).median().alias(f"{col}_median")
            ])

            # Join the median values back to the original DataFrame
            self.df = self.df.join(median_values, on="symbol_id", how="left")

            # Find the first non-NaN value's row_id for each symbol_id group
            first_non_nan_row_id = (
                self.df.filter(pl.col(col).is_not_null())
                .group_by("symbol_id")
                .agg([pl.col("row_id").min().alias("first_non_nan_row_id")])
            )

            # Join the first_non_nan_row_id with the original dataframe
            self.df = self.df.join(first_non_nan_row_id, on="symbol_id", how="left")

            # Fill NaN values only until the first non-NaN row
            self.df = self.df.with_columns([
                pl.when(pl.col(col).is_null() & (pl.col("row_id") < pl.col("first_non_nan_row_id")))
                    .then(pl.col(f"{col}_median"))
                    .otherwise(pl.col(col))  # Keep original value after first non-NaN
                .alias(col)
            ])
            
            self.df = self.df.drop("first_non_nan_row_id", f"{col}_median")
        
        print("Dropping unnecessary columns...")
        self.df = self.df.drop(["row_id"])

        print("Checking columns...")
        print(self.df.columns)
       
        # Iterate over the columns with NaN values
        for col in self.nan_all_columns:
            print(f"Filling the rest NaNs with forward fill for {col}...")
            self.df = self.df.with_columns(
                pl.col(col).fill_null(strategy="forward").over("symbol_id").alias(col)
            )
    
    def _add_lags_stats(self):
        print("Adding lag stats...")
        lag_stats_df = self.df.select("date_id").unique()
        resp_cols = [f"responder_{idx}" for idx in range(9)]

        for col in resp_cols:
            print(f"Processing {col}...")
            min_df = self.df.group_by("date_id").agg(pl.col(col).min().alias(f"{col}_lag_1_min"))
            max_df = self.df.group_by("date_id").agg(pl.col(col).max().alias(f"{col}_lag_1_max"))
            mean_df = self.df.group_by("date_id").agg(pl.col(col).mean().alias(f"{col}_lag_1_mean"))
            sum_df = self.df.group_by("date_id").agg(pl.col(col).sum().alias(f"{col}_lag_1_sum"))

            lag_stats_df = lag_stats_df.join(min_df, on="date_id", how="left")
            lag_stats_df = lag_stats_df.join(max_df, on="date_id", how="left")
            lag_stats_df = lag_stats_df.join(mean_df, on="date_id", how="left")
            lag_stats_df = lag_stats_df.join(sum_df, on="date_id", how="left")

        # add all date_id by 1 and join to df
        lag_stats_df = lag_stats_df.with_columns(pl.col("date_id") + 1)
        self.df = self.df.join(lag_stats_df, on="date_id", how="left")

    def save_categorical_mapping(self):
        print("Mapping categorical values to integers...")

        for col in self.categorical_columns:
            unique_values = self.df.select(pl.col(col)).unique().to_numpy().flatten().astype(int).tolist()
            print("unique_values", unique_values)
            value_to_int = {val: idx + 1 for idx, val in enumerate(unique_values)} # 0 is reserved for new values
            print("value_to_int", value_to_int)

            cat_mapping_df = pl.DataFrame({
                f"{col}": list(value_to_int.keys()),
                f"mapped_{col}": list(value_to_int.values())
            })
            cat_mapping_df.write_parquet(os.path.join(self.out_dir, f"{col}_cat_mapping.parquet"))
            print(f"Categorical values {col} mapped and saved as {col}_cat_mapping.parquet.")
            
    def save_stats_dataframe(self):
        print("Saving feature stats as parquet files...")

        # Compute the mean, std, and median for each feature (instead of per symbol_id)
        # This will be useful for normalizing the features in the neural network
        stats_list = []
        # Iterate through the columns to compute the stats for each feature
        for col in self.all_columns:
            if col == "weight" or "feature" in col:
                print(f"Processing {col}...")

                mean_val = float(self.df.select(pl.col(col).mean().alias(f"{col}_mean")).to_numpy().flatten()[0])
                std_val = float(self.df.select(pl.col(col).std().alias(f"{col}_std")).to_numpy().flatten()[0])
                median_val = float(self.df.select(pl.col(col).median().alias(f"{col}_median")).to_numpy().flatten()[0])

                stats_list.append({
                    "feature": col,
                    "mean": mean_val,
                    "std": std_val,
                    "median": median_val
                })

        # Convert the stats list to a DataFrame
        stats_df = pl.DataFrame(stats_list)
        stats_df.write_parquet(os.path.join(self.out_dir, "stats.parquet"))

        '''
        all_stats_df = self.df.select("symbol_id").unique() 
        # Iterate through columns to compute the stats for each feature
        for col in self.all_columns:
            if col == "weight" or "feature" in col:
                print(f"Processing {col}...")
                mean_df = self.df.group_by("symbol_id").agg(pl.col(col).mean().alias(f"{col}_mean"))
                std_df = self.df.group_by("symbol_id").agg(pl.col(col).std().alias(f"{col}_std"))
                median_df = self.df.group_by("symbol_id").agg(pl.col(col).median().alias(f"{col}_median"))
                all_stats_df = all_stats_df.join(mean_df, on="symbol_id", how="left")
                all_stats_df = all_stats_df.join(std_df, on="symbol_id", how="left")
                all_stats_df = all_stats_df.join(median_df, on="symbol_id", how="left")
        all_stats_df.write_parquet(os.path.join(self.out_dir, "stats.parquet"))
        '''

    def save_train_val_dataframe(self):
        print("Saving preprocessed data as training/validation parquet files...")

        print(f"training covering date_id's:[{self.train_start_date}, {self.train_end_date}]")
        train_save_path = os.path.join(self.out_dir, "training.parquet")
        df = self.df.filter(
            pl.col("date_id").ge(self.train_start_date) & pl.col("date_id").le(self.train_end_date)
        )
        df.write_parquet(train_save_path, partition_by=["date_id"])

        print(f"validation covering date_id's:[{self.val_start_date}, {self.val_end_date}]")
        val_save_path = os.path.join(self.out_dir, "validation.parquet")
        df = self.df.filter(
            pl.col("date_id").ge(self.val_start_date) & pl.col("date_id").le(self.val_end_date)
        )
        df.write_parquet(val_save_path, partition_by=["date_id"])

    def save_full_dataframe_debug(self):
        print("[Debug] Saving full preprocessed data as parquet files...")
        save_path = os.path.join(self.out_dir, "full.parquet")
        self.df.write_parquet(save_path, partition_by=["date_id"])


if __name__ == "__main__":
    train_file = "./dataset/train.parquet"
    train_start_date = 685
    train_end_date = 1484
    val_start_date = 1500
    val_end_date = 1699
    out_dir = "./preprocessed_dataset"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    dp = DataPreprocess(train_file, train_start_date, train_end_date, val_start_date, val_end_date, out_dir)
    #dp.save_categorical_mapping()
    dp.save_stats_dataframe()
    dp.save_train_val_dataframe()

# TODO: Dataset module
# 1. Load mean.parquet and map_categorical.parquet.
# 2. Dynamically read training.parquet files. 
# 3. Normalize features with tanh, except for id=9,10,11 and responders. 
#   - Tanh( [x-mean(x)] * 1/sigma(x) )
# 4. Normalize responders by dividing by 5.
# 5. For id=9,10,11, use the integer values from map_categorical.parquet.
# 6. Train/val Split, set a gap between train and val. 
#   - Make sure always past for training, and future for validation.

'''
train = pl.scan_parquet("dataset/train.parquet")
res = train.group_by("date_id").agg(
    pl.col("time_id").count().alias("num_timesteps")
).sort(by=['date_id']).collect()

last_date_id = res.select("date_id").row(-1)[0]

# Get the top date_id after sorting
sorted_date_id_by_completeness = res.sort(by=["num_timesteps", "date_id"], descending=[True, False])
first_complete_date_id = sorted_date_id_by_completeness.select("date_id").row(0)[0]

# Do 70/30 split between two date_ids
split_date_id = int(first_complete_date_id + (last_date_id - first_complete_date_id) * 0.7)
'''

