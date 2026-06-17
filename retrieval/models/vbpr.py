"""
VBPR (Visual Bayesian Personalized Ranking) model implementation.

Based on the paper "VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback"
and the implementation at https://github.com/aaossa/VBPR-PyTorch

Reference:
    @inproceedings{he2016vbpr,
        title={VBPR: visual bayesian personalized ranking from implicit feedback},
        author={He, Ruining and McAuley, Julian},
        booktitle={Proceedings of the AAAI conference on artificial intelligence},
        volume={30},
        number={1},
        year={2016}
    }
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class VBPR(nn.Module):
    """
    VBPR model: Visual Bayesian Personalized Ranking.
    
    Combines traditional collaborative filtering with visual features.
    The model predicts user-item preference scores using:
    - User embeddings (gamma_user)
    - Item embeddings (gamma_item)
    - Visual features (theta_item) projected through a CNN layer
    
    Architecture:
        score(u, i) = gamma_user[u]^T * gamma_item[i] + 
                      gamma_user[u]^T * (E * theta_item[i])
    
    Where:
        - gamma_user: User latent factors [n_users, dim_gamma]
        - gamma_item: Item latent factors [n_items, dim_gamma]
        - theta_item: Visual features [n_items, visual_dim]
        - E: Projection matrix [dim_theta, dim_gamma]
    """
    
    def __init__(
        self,
        n_users: int,
        n_items: int,
        visual_features: torch.Tensor,
        dim_gamma: int = 20,
        dim_theta: int = 20,
        visual_dim: Optional[int] = None,
    ):
        """
        Args:
            n_users: Number of users
            n_items: Number of items
            visual_features: Pre-computed visual features [n_items, visual_dim]
            dim_gamma: Dimension of user/item latent factors (default: 20)
            dim_theta: Dimension of visual projection (default: 20)
            visual_dim: Dimension of visual features (auto-detected if None)
        """
        super(VBPR, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.dim_gamma = dim_gamma
        self.dim_theta = dim_theta
        
        # Visual features: [n_items, visual_dim]
        if visual_features.dim() == 2:
            self.register_buffer('visual_features', visual_features)
            self.visual_dim = visual_features.size(1)
        else:
            raise ValueError(f"visual_features must be 2D tensor, got {visual_features.dim()}D")
        
        if visual_dim is not None and visual_dim != self.visual_dim:
            raise ValueError(
                f"visual_dim mismatch: provided {visual_dim}, "
                f"but visual_features has {self.visual_dim}"
            )
        
        # User latent factors: [n_users, dim_gamma]
        self.gamma_user = nn.Embedding(n_users, dim_gamma)
        
        # Item latent factors: [n_items, dim_gamma]
        self.gamma_item = nn.Embedding(n_items, dim_gamma)
        
        # Visual projection matrix: [visual_dim, dim_theta]
        # Projects visual features to dim_theta space
        self.E = nn.Linear(self.visual_dim, dim_theta, bias=False)
        
        # User visual preference: [n_users, dim_theta]
        # Captures user's preference for visual aspects
        self.theta_user = nn.Embedding(n_users, dim_theta)
        
        # Initialize embeddings
        self._init_embeddings()
    
    def _init_embeddings(self):
        """Initialize embeddings with Xavier normal for better convergence.
        
        Note: Original VBPR uses std=0.01, but this can be too small and lead to
        slow learning. Xavier initialization is more standard for recommendation models.
        """
        # Use Xavier normal for embeddings (better for matrix factorization models)
        nn.init.xavier_normal_(self.gamma_user.weight)
        nn.init.xavier_normal_(self.gamma_item.weight)
        nn.init.xavier_normal_(self.theta_user.weight)
        # For projection matrix E, use smaller initialization to match visual feature scale
        nn.init.normal_(self.E.weight, mean=0.0, std=0.1)
    
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
        # Get embeddings
        gamma_u = self.gamma_user(user_ids)  # [batch_size, dim_gamma]
        gamma_i = self.gamma_item(item_ids)  # [batch_size, dim_gamma]
        
        # Get visual features for items
        visual_i = self.visual_features[item_ids]  # [batch_size, visual_dim]
        
        # Project visual features: E * visual_i
        projected_visual = self.E(visual_i)  # [batch_size, dim_theta]
        
        # Get user visual preference
        theta_u = self.theta_user(user_ids)  # [batch_size, dim_theta]
        
        # Compute scores:
        # score = gamma_u^T * gamma_i + theta_u^T * (E * visual_i)
        cf_score = torch.sum(gamma_u * gamma_i, dim=1)  # [batch_size]
        visual_score = torch.sum(theta_u * projected_visual, dim=1)  # [batch_size]
        
        scores = cf_score + visual_score  # [batch_size]
        
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
            # Get user embeddings
            gamma_u = self.gamma_user.weight[user_id:user_id+1]  # [1, dim_gamma]
            theta_u = self.theta_user.weight[user_id:user_id+1]  # [1, dim_theta]
            
            # Get all item embeddings
            gamma_i = self.gamma_item.weight  # [n_items, dim_gamma]
            
            # Get all visual features
            visual_i = self.visual_features  # [n_items, visual_dim]
            projected_visual = self.E(visual_i)  # [n_items, dim_theta]
            
            # Compute scores
            cf_scores = torch.matmul(gamma_u, gamma_i.t()).squeeze(0)  # [n_items]
            visual_scores = torch.matmul(theta_u, projected_visual.t()).squeeze(0)  # [n_items]
            
            scores = cf_scores + visual_scores  # [n_items]
            
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
            gamma_u = self.gamma_user(user_ids)  # [batch_size, dim_gamma]
            theta_u = self.theta_user(user_ids)  # [batch_size, dim_theta]
            
            # Get all item embeddings
            gamma_i = self.gamma_item.weight  # [n_items, dim_gamma]
            
            # Get all visual features
            visual_i = self.visual_features  # [n_items, visual_dim]
            projected_visual = self.E(visual_i)  # [n_items, dim_theta]
            
            # Compute scores for all users at once
            cf_scores = torch.matmul(gamma_u, gamma_i.t())  # [batch_size, n_items]
            visual_scores = torch.matmul(theta_u, projected_visual.t())  # [batch_size, n_items]
            
            scores = cf_scores + visual_scores  # [batch_size, n_items]
            
        return scores
    
    def loss(
        self,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: torch.Tensor,
        lambda_reg: float = 0.01
    ) -> torch.Tensor:
        """
        Compute BPR loss with regularization.
        
        Args:
            user_ids: User IDs [batch_size]
            pos_item_ids: Positive item IDs [batch_size]
            neg_item_ids: Negative item IDs [batch_size]
            lambda_reg: Regularization weight (default: 0.01)
            
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
        gamma_u = self.gamma_user(user_ids)  # [batch_size, dim_gamma]
        gamma_i_pos = self.gamma_item(pos_item_ids)  # [batch_size, dim_gamma]
        gamma_i_neg = self.gamma_item(neg_item_ids)  # [batch_size, dim_gamma]
        theta_u = self.theta_user(user_ids)  # [batch_size, dim_theta]
        
        # Normalize regularization by batch size to keep it in same scale as BPR loss
        batch_size = user_ids.size(0)
        # Regularize embeddings (user, item, theta_user)
        embedding_reg = (
            torch.sum(gamma_u ** 2) +
            torch.sum(gamma_i_pos ** 2) +
            torch.sum(gamma_i_neg ** 2) +
            torch.sum(theta_u ** 2)
        ) / batch_size
        
        # Regularize projection matrix E (important: E can grow very large without regularization)
        # Use a smaller weight for E since it's a global parameter (not per-sample)
        E_reg = torch.sum(self.E.weight ** 2) / (self.E.weight.numel() / batch_size) if batch_size > 0 else torch.sum(self.E.weight ** 2)
        
        reg_loss = lambda_reg * (embedding_reg + 0.1 * E_reg)  # E regularization with smaller weight
        
        total_loss = bpr_loss + reg_loss
        
        return total_loss

