import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)  

        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits
    
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, emb, mask=None):
        vsum = torch.mm(mask, emb)
        row_sum = torch.sum(mask, 1)
        row_sum = row_sum.expand((vsum.shape[1], row_sum.shape[0])).T
        global_emb = vsum / row_sum 
          
        return F.normalize(global_emb, p=2, dim=1) 

class AttentionLayer(Module):
    """
    Attention layer for combining spatial and feature graph embeddings.
    """
    def __init__(self, in_feat, out_feat):
        super(AttentionLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        
        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)
        
    def forward(self, emb_spatial, emb_feat):
        emb = []
        emb.append(torch.unsqueeze(torch.squeeze(emb_spatial), dim=1))
        emb.append(torch.unsqueeze(torch.squeeze(emb_feat), dim=1))
        self.emb = torch.cat(emb, dim=1)
        
        self.v = F.tanh(torch.matmul(self.emb, self.w_omega))
        self.vu = torch.matmul(self.v, self.u_omega)
        self.alpha = F.softmax(torch.squeeze(self.vu) + 1e-6)  
        
        emb_combined = torch.matmul(torch.transpose(self.emb, 1, 2), torch.unsqueeze(self.alpha, -1))
    
        return torch.squeeze(emb_combined), self.alpha
    

class MultiHeadAttentionLayer(Module):
    """
    Multi-head attention layer for combining spatial and feature graph embeddings.
    """
    def __init__(self, in_feat, out_feat, num_heads=4, dropout=0.1,):
        super(MultiHeadAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.out_feat = out_feat
        self.head_dim = out_feat // num_heads  # Dimension of each head
        assert self.head_dim * num_heads == out_feat, "out_feat must be divisible by num_heads"

        # Create independent AttentionLayer for each head
        self.attention_heads = nn.ModuleList([
            AttentionLayer(in_feat, self.head_dim,) for _ in range(num_heads)
        ])

        # Final linear layer, input dimension should be head_dim * num_heads
        self.fc = nn.Linear(self.out_feat * num_heads, out_feat)
        self.dropout = dropout

    def forward(self, emb_spatial, emb_feat):
        # Attention outputs for each head
        attn_outputs = [head(emb_spatial, emb_feat)[0] for head in self.attention_heads]
        
        # Concatenate outputs from each head on the last dimension
        concat_attn = torch.cat(attn_outputs, dim=-1)
        
        # Apply linear transformation and dropout to the concatenated result
        emb_combined = self.fc(F.dropout(concat_attn, p=self.dropout, training=self.training))
        
        return emb_combined

class Encoder(Module):
    def __init__(self, in_features, out_features, graph_neigh, dropout=0.1, act=F.relu, num_heads=4):
        super(Encoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.graph_neigh = graph_neigh
        self.dropout = dropout
        self.act = act
        
        self.spatial_weight = nn.Parameter(torch.tensor(1.0))
        self.feature_weight = nn.Parameter(torch.tensor(1.0))
        
        self.weight_spatial = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.weight_feat = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.weight_back = Parameter(torch.FloatTensor(self.out_features, self.in_features))  # Weight for backward propagation
        self.reset_parameters()
        
        self.disc = Discriminator(self.out_features)
        self.attention_layer = MultiHeadAttentionLayer(self.out_features, self.out_features, num_heads, dropout)  

        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_spatial)
        torch.nn.init.xavier_uniform_(self.weight_feat)
        torch.nn.init.xavier_uniform_(self.weight_back)

    def forward(self, feat, feat_a, adj, adj_feat):
        # Spatial graph convolution and weighting
        z_spatial = F.dropout(feat, self.dropout, self.training)
        z_spatial = torch.mm(z_spatial, self.weight_spatial)
        z_spatial = torch.spmm(adj, z_spatial)
        z_spatial = z_spatial * self.spatial_weight

        # Feature graph convolution and weighting
        z_feat = F.dropout(feat, self.dropout, self.training)
        z_feat = torch.mm(z_feat, self.weight_feat)
        z_feat = torch.spmm(adj_feat, z_feat)
        z_feat = z_feat * self.feature_weight

        # Combine spatial and feature embeddings using attention mechanism
        emb_combined = self.attention_layer(z_spatial, z_feat)

        # h: Map embedding back to original feature space
        h = torch.mm(emb_combined, self.weight_back)

        # Keep the embedding representation after convolution
        hiden_emb = emb_combined

        # Contrastive learning part
        z_a_spatial = F.dropout(feat_a, self.dropout, self.training)
        z_a_spatial = torch.mm(z_a_spatial, self.weight_spatial)
        z_a_spatial = torch.spmm(adj, z_a_spatial)
        z_a_spatial = z_a_spatial * self.spatial_weight

        z_a_feat = F.dropout(feat_a, self.dropout, self.training)
        z_a_feat = torch.mm(z_a_feat, self.weight_feat)
        z_a_feat = torch.spmm(adj_feat, z_a_feat)
        z_a_feat = z_a_feat * self.feature_weight

        emb_a_combined = self.attention_layer(z_a_spatial, z_a_feat)

        g = self.read(emb_combined, self.graph_neigh)
        g = self.sigm(g)

        g_a = self.read(emb_a_combined, self.graph_neigh)
        g_a = self.sigm(g_a)

        ret = self.disc(g, emb_combined, emb_a_combined)
        ret_a = self.disc(g_a, emb_a_combined, emb_combined)

        # Return hiden_emb, h, ret and ret_a
        return hiden_emb, h, ret, ret_a

class Encoder_sparse(Module):
    """
    Sparse version of Encoder with Multi-Head Attention mechanism and contrastive learning.
    """
    def __init__(self, in_features, out_features, graph_neigh, dropout=0.1, act=F.relu, num_heads=4):
        super(Encoder_sparse, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.graph_neigh = graph_neigh
        self.dropout = dropout
        self.act = act

        # Initialize weights
        self.spatial_weight = nn.Parameter(torch.tensor(1.0))  # Weight coefficient for spatial graph
        self.feature_weight = nn.Parameter(torch.tensor(1.0))  # Weight coefficient for feature graph

        # Parameter matrices
        self.weight_spatial = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.weight_feat = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.weight_back = Parameter(torch.FloatTensor(self.out_features, self.in_features))  # Weight for backward propagation
        self.reset_parameters()

        # Use multi-head attention mechanism and retain contrastive learning module
        self.disc = Discriminator(self.out_features)
        self.attention_layer = MultiHeadAttentionLayer(self.out_features, self.out_features, num_heads, dropout)

        # Sigmoid and Readout modules
        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_spatial)
        torch.nn.init.xavier_uniform_(self.weight_feat)
        torch.nn.init.xavier_uniform_(self.weight_back)

    def forward(self, feat, feat_a, adj, adj_feat):
        # 1. Spatial graph convolution and weighting
        z_spatial = F.dropout(feat, self.dropout, self.training)
        z_spatial = torch.spmm(adj, torch.mm(z_spatial, self.weight_spatial))  # Sparse adjacency matrix operation
        z_spatial = z_spatial * self.spatial_weight

        # 2. Feature graph convolution and weighting
        z_feat = F.dropout(feat, self.dropout, self.training)
        z_feat = torch.spmm(adj_feat, torch.mm(z_feat, self.weight_feat))  # Sparse adjacency matrix operation
        z_feat = z_feat * self.feature_weight

        # 3. Combine spatial and feature embeddings using multi-head attention mechanism
        emb_combined = self.attention_layer(z_spatial, z_feat)

        # 4. Backward propagation to original feature space
        h = torch.mm(emb_combined, self.weight_back)

        # Save embedding representation after convolution
        hiden_emb = emb_combined

        # Contrastive learning part: Compare feature graph and spatial graph embeddings
        z_a_spatial = F.dropout(feat_a, self.dropout, self.training)
        z_a_spatial = torch.spmm(adj, torch.mm(z_a_spatial, self.weight_spatial))
        z_a_spatial = z_a_spatial * self.spatial_weight

        z_a_feat = F.dropout(feat_a, self.dropout, self.training)
        z_a_feat = torch.spmm(adj_feat, torch.mm(z_a_feat, self.weight_feat))
        z_a_feat = z_a_feat * self.feature_weight

        # Combine outputs using attention layer
        emb_a_combined = self.attention_layer(z_a_spatial, z_a_feat)

        # Extract global representation using AvgReadout
        g = self.read(emb_combined, self.graph_neigh)
        g = self.sigm(g)

        g_a = self.read(emb_a_combined, self.graph_neigh)
        g_a = self.sigm(g_a)

        # Calculate contrastive learning loss
        ret = self.disc(g, emb_combined, emb_a_combined)
        ret_a = self.disc(g_a, emb_a_combined, emb_combined)

        # Return calculation results: hiden_emb, h, ret, ret_a
        return hiden_emb, h, ret, ret_a
   

class Encoder_sc(torch.nn.Module):
    def __init__(self, dim_input, dim_output, dropout=0.0, act=F.relu):
        super(Encoder_sc, self).__init__()
        # Define encoder part
        self.encoder_dim1 = 256
        self.encoder_dim2 = 64
        self.encoder_dim3 = 32
        self.act = act
        self.dropout = dropout

        # Encoder: Dimensionality reduction from input dimension to latent space
        self.encoder_fc1 = torch.nn.Linear(dim_input, self.encoder_dim1)
        self.encoder_fc2 = torch.nn.Linear(self.encoder_dim1, self.encoder_dim2)
        self.encoder_fc3 = torch.nn.Linear(self.encoder_dim2, self.encoder_dim3)

        # Decoder: Dimensionality increase from latent space back to output dimension
        self.decoder_fc1 = torch.nn.Linear(self.encoder_dim3, self.encoder_dim2)
        self.decoder_fc2 = torch.nn.Linear(self.encoder_dim2, self.encoder_dim1)
        self.decoder_fc3 = torch.nn.Linear(self.encoder_dim1, dim_input)

    def forward(self, x):
        # Encoding process
        x = F.dropout(x, self.dropout, self.training)
        x = self.act(self.encoder_fc1(x))
        
        x = F.dropout(x, self.dropout, self.training)
        x = self.act(self.encoder_fc2(x))

        x = F.dropout(x, self.dropout, self.training)
        x = self.act(self.encoder_fc3(x))  # Encoder output to latent space
        
        # Decoding process
        x = F.dropout(x, self.dropout, self.training)
        x = self.act(self.decoder_fc1(x))
        
        x = F.dropout(x, self.dropout, self.training)
        x = self.act(self.decoder_fc2(x))

        x = F.dropout(x, self.dropout, self.training)
        x = self.decoder_fc3(x)  # Decoder output back to original dimension
        
        return x

class Encoder_map(torch.nn.Module):
    def __init__(self, n_cell, n_spot):
        super(Encoder_map, self).__init__()
        self.n_cell = n_cell
        self.n_spot = n_spot
          
        self.M = Parameter(torch.FloatTensor(self.n_cell, self.n_spot))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.M)
        
    def forward(self):
        x = self.M
        
        return x 
