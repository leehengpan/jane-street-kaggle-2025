import polars as pl
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
import polars as pl


class CustomDataset(Dataset):
    def __init__(self, data_path, cat_mapping_paths, stats_path):
        assert len(cat_mapping_paths) == 3
        # 22+1 + 9+1 + 31+1 = 65
        self.cat_features = ["feature_09", "feature_10", "feature_11"] 
        # 36 + 1 + 76 = 113
        self.cont_features = ["weight"] + \
                             [f"feature_{i:02d}" for i in range(79) if f"feature_{i:02d}" not in self.cat_features] + \
                             [f"responder_{i}_lag_1_min" for i in range(9)] + \
                             [f"responder_{i}_lag_1_max" for i in range(9)] + \
                             [f"responder_{i}_lag_1_mean" for i in range(9)] + \
                             [f"responder_{i}_lag_1_sum" for i in range(9)]
        self.label = "responder_6"
        self.stats = pl.read_parquet(stats_path)

        cat_mappings = {feature: pl.read_parquet(cat_mapping_paths[feature]) for feature in self.cat_features}
        self.cat_mapping = {
            feature: dict(zip(mappings[feature], mappings[f"mapped_{feature}"])) for feature, mappings in cat_mappings.items()
        }
        self.df = pl.read_parquet(data_path)
        self.df = self.df.filter(pl.col("date_id") > 1100)
        # self._normalize_df()
        
    def __len__(self):
        return len(self.df)

    def _normalize_df(self):
        for feature in self.cont_features:
            '''
            # tanh normalization, using std as sensitivity
            if "feature" in feature or "weight" in feature:
                mean = self.stats.filter(pl.col("feature") == feature)["mean"].item()
                std = self.stats.filter(pl.col("feature") == feature)["std"].item()
                self.df = self.df.with_columns(
                    np.tanh((pl.col(feature) - mean) / std)
                )
            '''

            # normalize: rescale by mean and std (z score)
            if "feature" in feature or "weight" in feature:
                mean = self.stats.filter(pl.col("feature") == feature)["mean"].item()
                std = self.stats.filter(pl.col("feature") == feature)["std"].item()
                self.df = self.df.with_columns(
                    (pl.col(feature) - mean) / std
                )

            # normalize lag features dividing by 5
            else:
                self.df = self.df.with_columns(
                    pl.col(feature) / 5
                )
                
        # normalize label dividing by 5
        self.df = self.df.with_columns(pl.col(self.label)/5)

    def __getitem__(self, idx):
        raw = self.df[idx]
        x = torch.tensor(raw[self.cont_features].to_numpy(), dtype=torch.float32).squeeze()
        c = [self.cat_mapping[feature][raw[feature].item()] for feature in self.cat_features]
        c = torch.tensor(c, dtype=torch.int64)
        y = torch.tensor(raw[self.label].item(), dtype=torch.float32)
        return x,c,y
    
# Usage example:
if __name__ == '__main__':
    data_path = "../preprocessed_dataset/validation.parquet"
    cat_mapping_paths = {f"feature_{i:02d}" : f"../preprocessed_dataset/feature_{i:02d}_cat_mapping.parquet" for i in range(9,12)}
    stats_path = "../preprocessed_dataset/stats.parquet"
    dataset = CustomDataset(data_path, cat_mapping_paths, stats_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for x, c, y in dataloader:
        print(x.shape, c.shape, y.shape) # (32,113) (32,3) (32)
        break






'''
print("Preparing data...")

input_path = './'
TRAINING = True
feature_names = [f"feature_{i:02d}" for i in range(79)] + [f"responder_{idx}_lag_1" for idx in range(9)]
label_name = 'responder_6'
weight_name = 'weight'
train_name = "nn_input_df_with_lags.pickle"
valid_name = "nn_valid_df_with_lags.pickle"

if TRAINING and not os.path.exists(train_name):
    df = pl.scan_parquet(f"{input_path}/training.parquet").collect().to_pandas()
    valid = pl.scan_parquet(f"{input_path}/validation.parquet").collect().to_pandas()
    # df = pd.concat([df, valid]).reset_index(drop=True) # A trick to boost LB from 0.0045->0.005
    with open(train_name, "wb") as w:
        pickle.dump(df, w)
    with open(valid_name, "wb") as w:
        pickle.dump(valid, w)
elif TRAINING:
    with open(train_name, "rb") as r:
        df = pickle.load(r)
    with open(valid_name, "rb") as r:
        valid = pickle.load(r)

X_train = df[ feature_names ]
y_train = df[ label_name ]
w_train = df[ "weight" ]
X_valid = valid[ feature_names ]
y_valid = valid[ label_name ]
w_valid = valid[ "weight" ]

import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import (LightningDataModule, LightningModule, Trainer)
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Timer
from pytorch_lightning.loggers import WandbLogger
import wandb
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# prompt: check if torch cuda exists
print("is torch cuda available ",torch.cuda.is_available())


class custom_args():
    def __init__(self):
        self.usegpu = True
        self.gpuid = 0
        self.seed = 42
        self.model = 'nn'
        self.use_wandb = True
        self.project = 'js-xs-nn-with-lags'
        self.dname = "./"
        self.loader_workers = 4
        self.bs = 8192
        self.lr = 1e-3
        self.weight_decay = 5e-4
        self.dropouts = [0.1, 0.1]
        self.n_hidden = [512, 512, 256]
        self.patience = 25
        self.max_epochs = 2000
        self.N_fold = 5

my_args = custom_args()

class CustomDataset(Dataset):
    def __init__(self, df, accelerator):
        self.features = torch.FloatTensor(df[feature_names].values).to(accelerator)
        self.labels = torch.FloatTensor(df[label_name].values).to(accelerator)
        self.weights = torch.FloatTensor(df[weight_name].values).to(accelerator)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        w = self.weights[idx]
        return x, y, w


class DataModule(LightningDataModule):
    def __init__(self, train_df, batch_size, valid_df=None, accelerator='cpu'):
        super().__init__()
        self.df = train_df
        self.batch_size = batch_size
        self.dates = self.df['date_id'].unique()
        self.accelerator = accelerator
        self.train_dataset = None
        self.valid_df = None
        if valid_df is not None:
            self.valid_df = valid_df
        self.val_dataset = None

    def setup(self, fold=0, N_fold=5, stage=None):
        # Split dataset
        selected_dates = [date for ii, date in enumerate(self.dates) if ii % N_fold != fold]
        df_train = self.df.loc[self.df['date_id'].isin(selected_dates)]
        self.train_dataset = CustomDataset(df_train, self.accelerator)
        if self.valid_df is not None:
            df_valid = self.valid_df
            self.val_dataset = CustomDataset(df_valid, self.accelerator)

    def train_dataloader(self, n_workers=0):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=n_workers)

    def val_dataloader(self, n_workers=0):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=n_workers)

# Custom R2 metric for validation
def r2_val(y_true, y_pred, sample_weight):
    r2 = 1 - np.average((y_pred - y_true) ** 2, weights=sample_weight) / (np.average((y_true) ** 2, weights=sample_weight) + 1e-38)
    return r2

class NN(LightningModule):
    def __init__(self, input_dim, hidden_dims, dropouts, lr, weight_decay):
        super().__init__()
        self.save_hyperparameters()
        layers = []
        in_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.BatchNorm1d(in_dim))
            if i > 0:
                layers.append(nn.SiLU())
            if i < len(dropouts):
                layers.append(nn.Dropout(dropouts[i]))
            layers.append(nn.Linear(in_dim, hidden_dim))
            # layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1)) 
        layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)
        self.lr = lr
        self.weight_decay = weight_decay
        self.validation_step_outputs = []

    def forward(self, x):
        return 5 * self.model(x).squeeze(-1)  

    def training_step(self, batch):
        x, y, w = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y, reduction='none') * w  #
        loss = loss.mean()
        self.log('train_loss', loss, on_step=False, on_epoch=True, batch_size=x.size(0))
        return loss

    def validation_step(self, batch):
        x, y, w = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y, reduction='none') * w
        loss = loss.mean()
        self.log('val_loss', loss, on_step=False, on_epoch=True, batch_size=x.size(0))
        self.validation_step_outputs.append((y_hat, y, w))
        return loss

    def on_validation_epoch_end(self):
        """Calculate validation WRMSE at the end of the epoch."""
        y = torch.cat([x[1] for x in self.validation_step_outputs]).cpu().numpy()
        if self.trainer.sanity_checking:
            prob = torch.cat([x[0] for x in self.validation_step_outputs]).cpu().numpy()
        else:
            prob = torch.cat([x[0] for x in self.validation_step_outputs]).cpu().numpy()
            weights = torch.cat([x[2] for x in self.validation_step_outputs]).cpu().numpy()
            # r2_val
            val_r_square = r2_val(y, prob, weights)
            self.log("val_r_square", val_r_square, prog_bar=True, on_step=False, on_epoch=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5,
                                                               verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            }
        }

    def on_train_epoch_end(self):
        if self.trainer.sanity_checking:
            return
        epoch = self.trainer.current_epoch
        metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in self.trainer.logged_metrics.items()}
        formatted_metrics = {k: f"{v:.5f}" for k, v in metrics.items()}
        print(f"Epoch {epoch}: {formatted_metrics}")

print("Preparing arguments...")

args = my_args

# checking device
device = torch.device(f'cuda:{args.gpuid}' if torch.cuda.is_available() and args.usegpu else 'cpu')
accelerator = 'gpu' if torch.cuda.is_available() and args.usegpu else 'cpu'
loader_device = 'cpu'

# Initialize Data Module
df[feature_names] = df[feature_names].fillna(method = 'ffill').fillna(0)
valid[feature_names] = valid[feature_names].fillna(method = 'ffill').fillna(0)
data_module = DataModule(df, batch_size=args.bs, valid_df=valid, accelerator=loader_device)


print("Start training...")

import gc
del df
gc.collect()
pl.seed_everything(args.seed)
for fold in range(args.N_fold):
    data_module.setup(fold, args.N_fold)
    # Obtain input dimension
    input_dim = data_module.train_dataset.features.shape[1]
    # Initialize Model
    model = NN(
        input_dim=input_dim,
        hidden_dims=args.n_hidden,
        dropouts=args.dropouts,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    # Initialize Logger
    if args.use_wandb:
        wandb_run = wandb.init(project=args.project, config=vars(args), reinit=True)
        logger = WandbLogger(experiment=wandb_run)
    else:
        logger = None
    # Initialize Callbacks
    early_stopping = EarlyStopping('val_loss', patience=args.patience, mode='min', verbose=False)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, verbose=False, filename=f"./models/nn_{fold}.model") 
    timer = Timer()
    # Initialize Trainer
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=[args.gpuid] if args.usegpu else None,
        logger=logger,
        callbacks=[early_stopping, checkpoint_callback, timer],
        enable_progress_bar=True
    )
    # Start Training
    trainer.fit(model, data_module.train_dataloader(args.loader_workers), data_module.val_dataloader(args.loader_workers))
    # You can find trained best model in your local path
    print(f'Fold-{fold} Training completed in {timer.time_elapsed("train"):.2f}s')
'''