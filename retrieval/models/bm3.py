"""
BM3 (Bootstrap Latent Representations for Multi-modal Recommendation) model implementation.

Based on the paper "Bootstrap Latent Representations for Multi-Modal Recommendation" (WWW'23)
and the implementation at https://github.com/enoche/BM3

Reference:
    @inproceedings{zhou2023bootstrap,
        author = {Zhou, Xin and Zhou, Hongyu and Liu, Yong and Zeng, Zhiwei and 
                  Miao, Chunyan and Wang, Pengwei and You, Yuan and Jiang, Feijun},
        title = {Bootstrap Latent Representations for Multi-Modal Recommendation},
        booktitle = {Proceedings of the ACM Web Conference 2023},
        pages = {845--854},
        year = {2023}
    }
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BM3(nn.Module):
    """
    BM3 model: Bootstrap Latent Representations for Multi-modal Recommendation.
    
    BM3 uses a bootstrap mechanism to learn latent representations from multimodal data.
    The model combines:
    - User/item embeddings (collaborative filtering)
    - Visual features (image embeddings)
    - Text features (text embeddings)
    
    Architecture:
        - Bootstrap mechanism: learns representations by bootstrapping from multimodal features
        - Multi-layer MLP to fuse visual and text features
        - Final score = CF score + multimodal score
    
    Based on the implementation at https://github.com/enoche/BM3
    """
    
    def __init__(
        self,
        n_users: int,
        n_items: int,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        embed_dim: int = 64,
        layers: int = 1,
        dropout: float = 0.5,
        visual_dim: Optional[int] = None,
        text_dim: Optional[int] = None,
    ):
        """
        Args:
            n_users: Number of users
            n_items: Number of items
            visual_features: Pre-computed visual features [n_items, visual_dim]
            text_features: Pre-computed text features [n_items, text_dim]
            embed_dim: Embedding dimension for users/items (default: 64)
            layers: Number of MLP layers for feature fusion (default: 1)
            dropout: Dropout rate (default: 0.5)
            visual_dim: Dimension of visual features (auto-detected if None)
            text_dim: Dimension of text features (auto-detected if None)
        """
        super(BM3, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.layers = layers
        self.dropout = dropout
        
        # Visual features: [n_items, visual_dim]
        if visual_features.dim() == 2:
            self.register_buffer('visual_features', visual_features)
            self.visual_dim = visual_features.size(1)
        else:
            raise ValueError(f"visual_features must be 2D tensor, got {visual_features.dim()}D")
        
        # Text features: [n_items, text_dim]
        if text_features.dim() == 2:
            self.register_buffer('text_features', text_features)
            self.text_dim = text_features.size(1)
        else:
            raise ValueError(f"text_features must be 2D tensor, got {text_features.dim()}D")
        
        if visual_dim is not None and visual_dim != self.visual_dim:
            raise ValueError(
                f"visual_dim mismatch: provided {visual_dim}, "
                f"but visual_features has {self.visual_dim}"
            )
        
        if text_dim is not None and text_dim != self.text_dim:
            raise ValueError(
                f"text_dim mismatch: provided {text_dim}, "
                f"but text_features has {text_dim}"
            )
        
        # User and item embeddings (collaborative filtering)
        self.user_embedding = nn.Embedding(n_users, embed_dim)
        self.item_embedding = nn.Embedding(n_items, embed_dim)
        
        # Bootstrap mechanism: project multimodal features to embedding space
        # Visual projection
        self.visual_proj = nn.Linear(self.visual_dim, embed_dim, bias=False)
        
        # Text projection
        self.text_proj = nn.Linear(self.text_dim, embed_dim, bias=False)
        
        # Feature fusion MLP (bootstrap mechanism)
        # Input: concatenated visual + text features
        # Output: fused multimodal representation
        input_dim = embed_dim * 2  # visual_proj + text_proj
        self.fusion_layers = nn.ModuleList()
        
        for i in range(layers):
            if i == 0:
                self.fusion_layers.append(nn.Linear(input_dim, embed_dim))
            else:
                self.fusion_layers.append(nn.Linear(embed_dim, embed_dim))
        
        self.dropout_layer = nn.Dropout(dropout)
        
        # Initialize embeddings
        self._init_embeddings()
    
    def _init_embeddings(self):
        """Initialize embeddings with Xavier normal initialization for better convergence."""
        # Use Xavier normal for embeddings (standard for recommendation models)
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)
        nn.init.xavier_normal_(self.visual_proj.weight)
        nn.init.xavier_normal_(self.text_proj.weight)
        
        for layer in self.fusion_layers:
            nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
    
    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass: compute user-item preference scores.
        
        Args:
            user_ids: User IDs [batch_size]
            item_ids: Item IDs [batch_size]
            
        Returns:
            Preference scores [batch_size]
        """
        # Get user and item embeddings (CF component)
        user_emb = self.user_embedding(user_ids)  # [batch_size, embed_dim]
        item_emb = self.item_embedding(item_ids)  # [batch_size, embed_dim]
        
        # Get multimodal features for items
        visual_feat = self.visual_features[item_ids]  # [batch_size, visual_dim]
        text_feat = self.text_features[item_ids]      # [batch_size, text_dim]
        
        # Project to embedding space (bootstrap mechanism)
        visual_proj = self.visual_proj(visual_feat)  # [batch_size, embed_dim]
        text_proj = self.text_proj(text_feat)         # [batch_size, embed_dim]
        
        # Concatenate and fuse multimodal features
        multimodal_input = torch.cat([visual_proj, text_proj], dim=1)  # [batch_size, embed_dim*2]
        
        # Pass through fusion MLP
        x = multimodal_input
        for i, layer in enumerate(self.fusion_layers):
            x = layer(x)
            if i < len(self.fusion_layers) - 1:  # No activation after last layer
                x = F.relu(x)
                x = self.dropout_layer(x)
        
        multimodal_emb = x  # [batch_size, embed_dim]
        
        # Combine CF and multimodal scores
        # CF score: user_emb^T * item_emb
        cf_score = torch.sum(user_emb * item_emb, dim=1)  # [batch_size]
        
        # Multimodal score: user_emb^T * multimodal_emb
        multimodal_score = torch.sum(user_emb * multimodal_emb, dim=1)  # [batch_size]
        
        # Final score
        scores = cf_score + multimodal_score  # [batch_size]
        
        return scores
    
    def predict_all(self, user_id: int) -> torch.Tensor:
        """
        Predict scores for all items for a given user.
        
        Args:
            user_id: User ID (0-indexed)
            
        Returns:
            Scores for all items [n_items]
        """
        self.eval()
        with torch.no_grad():
            # Get user embedding
            user_emb = self.user_embedding.weight[user_id:user_id+1]  # [1, embed_dim]
            
            # Get all item embeddings
            item_emb = self.item_embedding.weight  # [n_items, embed_dim]
            
            # Get all multimodal features
            visual_feat = self.visual_features  # [n_items, visual_dim]
            text_feat = self.text_features      # [n_items, text_dim]
            
            # Project to embedding space
            visual_proj = self.visual_proj(visual_feat)  # [n_items, embed_dim]
            text_proj = self.text_proj(text_feat)        # [n_items, embed_dim]
            
            # Concatenate and fuse
            multimodal_input = torch.cat([visual_proj, text_proj], dim=1)  # [n_items, embed_dim*2]
            
            # Pass through fusion MLP
            x = multimodal_input
            for i, layer in enumerate(self.fusion_layers):
                x = layer(x)
                if i < len(self.fusion_layers) - 1:
                    x = F.relu(x)
                    x = self.dropout_layer(x)
            
            multimodal_emb = x  # [n_items, embed_dim]
            
            # Compute scores
            cf_scores = torch.matmul(user_emb, item_emb.t()).squeeze(0)  # [n_items]
            multimodal_scores = torch.matmul(user_emb, multimodal_emb.t()).squeeze(0)  # [n_items]
            
            scores = cf_scores + multimodal_scores  # [n_items]
            
        return scores
    
    def predict_batch(self, user_ids: torch.Tensor) -> torch.Tensor:
        """
        Predict scores for all items for a batch of users (optimized for batch evaluation).
        
        Args:
            user_ids: User IDs [batch_size] (0-indexed)
            
        Returns:
            Scores for all items for each user [batch_size, n_items]
        """
        self.eval()
        with torch.no_grad():
            # Get user embeddings
            user_emb = self.user_embedding(user_ids)  # [batch_size, embed_dim]
            
            # Get all item embeddings
            item_emb = self.item_embedding.weight  # [n_items, embed_dim]
            
            # Get all multimodal features
            visual_feat = self.visual_features  # [n_items, visual_dim]
            text_feat = self.text_features      # [n_items, text_dim]
            
            # Project to embedding space
            visual_proj = self.visual_proj(visual_feat)  # [n_items, embed_dim]
            text_proj = self.text_proj(text_feat)        # [n_items, embed_dim]
            
            # Concatenate and fuse
            multimodal_input = torch.cat([visual_proj, text_proj], dim=1)  # [n_items, embed_dim*2]
            
            # Pass through fusion MLP
            x = multimodal_input
            for i, layer in enumerate(self.fusion_layers):
                x = layer(x)
                if i < len(self.fusion_layers) - 1:
                    x = F.relu(x)
                    x = self.dropout_layer(x)
            
            multimodal_emb = x  # [n_items, embed_dim]
            
            # Compute scores for all users at once
            cf_scores = torch.matmul(user_emb, item_emb.t())  # [batch_size, n_items]
            multimodal_scores = torch.matmul(user_emb, multimodal_emb.t())  # [batch_size, n_items]
            
            scores = cf_scores + multimodal_scores  # [batch_size, n_items]
            
        return scores
    
    def loss(
        self,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: torch.Tensor,
        lambda_reg: float = 1e-4  # Changed default from 0.1 to 1e-4 (same as MMGCN)
    ) -> torch.Tensor:
        """
        Compute BPR loss with regularization.
        
        Args:
            user_ids: User IDs [batch_size]
            pos_item_ids: Positive item IDs [batch_size]
            neg_item_ids: Negative item IDs [batch_size]
            lambda_reg: Regularization weight (default: 1e-4, changed from 0.1)
            
        Returns:
            BPR loss value
        """
        # Get scores
        pos_scores = self.forward(user_ids, pos_item_ids)  # [batch_size]
        neg_scores = self.forward(user_ids, neg_item_ids)  # [batch_size]
        
        # BPR loss: -log(sigmoid(pos_score - neg_score))
        diff = pos_scores - neg_scores
        bpr_loss = -torch.log(torch.sigmoid(diff) + 1e-10).mean()
        
        # Regularization (normalized by batch size to match BPR loss scale)
        user_emb = self.user_embedding(user_ids)
        item_emb_pos = self.item_embedding(pos_item_ids)
        item_emb_neg = self.item_embedding(neg_item_ids)
        
        # Normalize regularization by batch size to keep it in same scale as BPR loss
        batch_size = user_ids.size(0)
        # Regularize embeddings (user, item)
        embedding_reg = (
            torch.sum(user_emb ** 2) +
            torch.sum(item_emb_pos ** 2) +
            torch.sum(item_emb_neg ** 2)
        ) / batch_size
        
        # Regularize projection matrices (visual_proj, text_proj) and fusion layers
        # These are global parameters, so use smaller weight
        proj_reg = (
            torch.sum(self.visual_proj.weight ** 2) +
            torch.sum(self.text_proj.weight ** 2)
        ) / (self.visual_proj.weight.numel() / batch_size) if batch_size > 0 else (
            torch.sum(self.visual_proj.weight ** 2) +
            torch.sum(self.text_proj.weight ** 2)
        )
        
        # Regularize fusion layers
        fusion_reg = sum(torch.sum(layer.weight ** 2) for layer in self.fusion_layers)
        fusion_reg = fusion_reg / (sum(layer.weight.numel() for layer in self.fusion_layers) / batch_size) if batch_size > 0 else fusion_reg
        
        reg_loss = lambda_reg * (embedding_reg + 0.1 * proj_reg + 0.1 * fusion_reg)
        
        total_loss = bpr_loss + reg_loss
        
        return total_loss

