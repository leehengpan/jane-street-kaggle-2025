import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, in_dim, head_count, dropout):
        super(Attention, self).__init__()
        self.head_count = head_count
        self.head_dim = in_dim // head_count

        # Linear projections for queries, keys, and values
        self.query = nn.Linear(in_dim, in_dim)
        self.key = nn.Linear(in_dim, in_dim)
        self.value = nn.Linear(in_dim, in_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(in_dim, in_dim)

    def forward(self, x):
        batch_size, seq_len, in_dim = x.size()
        
        # Linear projections for query, key, and value
        query = self.query(x).view(batch_size, seq_len, self.head_count, self.head_dim).transpose(1, 2)
        key = self.key(x).view(batch_size, seq_len, self.head_count, self.head_dim).transpose(1, 2)
        value = self.value(x).view(batch_size, seq_len, self.head_count, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / self.head_dim ** 0.5
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        # Apply attention to the values
        attn_output = torch.matmul(attn_probs, value).transpose(1, 2).contiguous().view(batch_size, seq_len, in_dim)
        output = self.out_proj(attn_output)
        
        return output

class ComplexMLPWithAttention(nn.Module):
    def __init__(self, feat_in_dims, cat_in_dims, hidden_layers, emb_dims, proj_dims, dropout=0.4, num_cat_features=3, attention_heads=4):
        super(ComplexMLPWithAttention, self).__init__()

        assert len(cat_in_dims) == num_cat_features
        self.feat_in_dims = feat_in_dims
        self.cat_in_dims = cat_in_dims
        self.hidden_layers = hidden_layers
        self.emb_dims = emb_dims
        self.proj_dims = proj_dims
        self.dropout = dropout
        self.attention_heads = attention_heads

        # Embedding layers for categorical features
        self.embeddings = nn.ModuleList([nn.Embedding(in_dims, emb_dims) for in_dims in cat_in_dims])

        # Embedding processing layers
        self.embedding_linear_layers = nn.Sequential(
            nn.Linear(emb_dims, emb_dims),
            nn.BatchNorm1d(emb_dims),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Continuous feature projection layers
        self.in_feat_linear_layers = nn.Sequential(
            nn.Linear(feat_in_dims, proj_dims),
            nn.BatchNorm1d(proj_dims),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Attention layer
        self.attention = Attention(in_dim=proj_dims + emb_dims, head_count=self.attention_heads, dropout=dropout)

        # Increased the number of hidden layers and added residual connections
        layers = []
        input_size = proj_dims + emb_dims  # Combined input size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            if input_size != hidden_size:
                layers.append(nn.Linear(input_size, hidden_size))
            input_size = hidden_size
        self.total_hidden_layers = len(layers)
        self.hidden_layers = nn.Sequential(*layers)

        # Output layer
        self.output_layer = nn.Linear(input_size, 1)

    def forward(self, cont_features, cat_features):
        # Embed the categorical features
        embs = torch.stack([embedding(cat_features[:, i]) for i, embedding in enumerate(self.embeddings)])
        embs = torch.sum(embs, dim=0)
        embs = self.embedding_linear_layers(embs)

        # Process the continuous features
        cont_features = self.in_feat_linear_layers(cont_features)

        # Concatenate continuous and embedded features
        x = torch.cat((cont_features, embs), dim=1)

        # Pass through attention layer
        x = self.attention(x.unsqueeze(1)).squeeze(1)  # (batch_size, seq_len, feature_dim)

        # Pass through the hidden layers with residual connections
        i=0
        while i < self.total_hidden_layers:
            hidden = self.hidden_layers[i](x)
            i+=1
            hidden = self.hidden_layers[i](hidden)
            i+=1
            hidden = self.hidden_layers[i](hidden)
            i+=1
            hidden = self.hidden_layers[i](hidden)
            i+=1            
            if hidden.size(1) != x.size(1):
                x = self.hidden_layers[i](x)
                i+=1
                # x = nn.Linear(x.size(1), hidden.size(1))(x)  # Project input to match output size
            x = hidden + x  # Residual connection

        return self.output_layer(x).squeeze()

if __name__ == '__main__':
    device = "cuda"
    num_obs = 8
    feat_in_dims = 113  # Number of continuous features
    cat_in_dims = [23, 10, 32]  # Number of categories for each categorical feature
    hidden_layers = [32, 32, 64, 64, 128, 128, 64, 64, 32, 32 ]  # Increased number of layers and units
    emb_dims = 8  # Increased embedding dimension
    proj_dims = 32  # Increased projected continuous feature dimension
    dropout = 0.4

    model = ComplexMLPWithAttention(feat_in_dims, cat_in_dims, hidden_layers, emb_dims, proj_dims, dropout=dropout).to(device)

    cont_features = torch.randn(8, feat_in_dims).to(device)  # Batch of 8 samples with continuous features

    # Batch of 8 samples with categorical features (for 3 categories)
    cat_feat09 = torch.randint(0, 23, (8, 1))
    cat_feat10 = torch.randint(0, 10, (8, 1))
    cat_feat11 = torch.randint(0, 32, (8, 1))
    cat_features = torch.cat((cat_feat09, cat_feat10, cat_feat11), dim=1).to(device)

    print(cont_features.shape)  # (8,113)
    print(cat_features.shape)  # (8,3)
    output = model(cont_features, cat_features)
    print(output.shape)  # (8)
