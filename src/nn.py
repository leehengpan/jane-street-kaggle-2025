import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, feat_in_dims, cat_in_dims, hidden_layers, emb_dims, proj_dims, dropout=0.4, num_cat_features=3):
        super(MLP, self).__init__()

        assert len(cat_in_dims) == num_cat_features
        self.feat_in_dims = feat_in_dims
        self.cat_in_dims = cat_in_dims
        self.hidden_layers = hidden_layers
        self.emb_dims = emb_dims
        self.proj_dims = proj_dims
        self.dropout = dropout

        self.embeddings = nn.ModuleList([nn.Embedding(in_dims, emb_dims) for in_dims in cat_in_dims])
        self.embedding_linear_layers = nn.Sequential(
            nn.Linear(emb_dims, emb_dims),
            nn.BatchNorm1d(emb_dims), # bn makes training more stable
            nn.LeakyReLU(),
            nn.Dropout(dropout) 
        )
        self.in_feat_linear_layers = nn.Sequential(
            nn.Linear(feat_in_dims, proj_dims),
            nn.BatchNorm1d(proj_dims), # bn makes training more stable
            nn.LeakyReLU(),
            nn.Dropout(dropout) 
        )
        
        layers = []
        input_size = proj_dims + emb_dims  # Combined input size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))  
            layers.append(nn.LeakyReLU())         
            layers.append(nn.Dropout(dropout))   
            input_size = hidden_size
        self.hidden_layers = nn.Sequential(*layers)        
        self.output_layer = nn.Linear(input_size, 1)
        
    def forward(self, cont_features, cat_features):
        # Embed the categorical features
        embs = torch.stack([embedding(cat_features[:, i]) for i, embedding in enumerate(self.embeddings)])
        embs = torch.sum(embs, dim=0)
        embs = self.embedding_linear_layers(embs)
        # Process the continuous features
        cont_features = self.in_feat_linear_layers(cont_features)
        # Concatenate cont_features and embs
        x = torch.cat((cont_features, embs), dim=1)
        return self.output_layer(self.hidden_layers(x)).squeeze()

if __name__ == '__main__':
    num_obs = 8
    feat_in_dims = 113  # Number of continuous features
    cat_in_dims = [23, 10, 32]  # Number of categories for each categorical feature
    hidden_layers = [64, 64, 128]  # Hidden layer sizes
    emb_dims = 16  # Embedding dimension for categorical features
    proj_dims = 64  # Projected continuous feature dimension
    output_size = 1  # Output size (e.g., regression task)

    model = MLP(feat_in_dims, cat_in_dims, hidden_layers, emb_dims, proj_dims)

    cont_features = torch.randn(8, feat_in_dims)  # Batch of 8 samples with continuous features

    # Batch of 8 samples with categorical features (for 3 categories)
    # Each categorical feature has 23, 10, and 32 categories, respectively
    cat_feat09 = torch.randint(0, 23, (8, 1))
    cat_feat10 = torch.randint(0, 10, (8, 1))
    cat_feat11 = torch.randint(0, 32, (8, 1))
    cat_features = torch.cat((cat_feat09, cat_feat10, cat_feat11), dim=1)

    print(cont_features.shape) # (8,113)
    print(cat_features.shape) # (8,3)
    output = model(cont_features, cat_features) 
    print(output.shape) # (8)

