"""Contrastive continual learning components for IDA-SPADE v6.

Components:
- TemporalAttention: single-head attention over entity history windows
- EntityFeatureBuffer: stores per-entity ECBA features across windows
- ProjectionHead: maps backbone features to contrastive embedding space
- PrototypeModule: EMA class centroids for reversal-robust decisions
- sup_con_loss: supervised contrastive loss (Khosla et al. 2020)
- ContrastiveContinualMLP: unified model replacing ContinualMLP
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque


class TemporalAttention(nn.Module):
    """Single-head attention for fusing entity features across time windows.

    Input: current features (N, M) + history sequence (N, L, M)
    Output: temporally contextualized features (N, M) via residual connection
    """

    def __init__(self, feat_dim):
        super().__init__()
        self.W_q = nn.Linear(feat_dim, feat_dim, bias=False)
        self.W_k = nn.Linear(feat_dim, feat_dim, bias=False)
        self.W_v = nn.Linear(feat_dim, feat_dim, bias=False)
        self.scale = feat_dim ** 0.5

    def forward(self, current, history_seq):
        """
        current: (N, M) - current window's entity features
        history_seq: (N, L, M) - L windows of history (including current as last)
        Returns: (N, M) - attention output + residual
        """
        Q = self.W_q(current).unsqueeze(1)           # (N, 1, M)
        K = self.W_k(history_seq)                     # (N, L, M)
        V = self.W_v(history_seq)                     # (N, L, M)
        attn = torch.softmax(
            torch.bmm(Q, K.transpose(1, 2)) / self.scale, dim=2)  # (N, 1, L)
        context = torch.bmm(attn, V).squeeze(1)      # (N, M)
        return context + current                      # residual


class EntityFeatureBuffer:
    """Stores per-entity ECBA features across windows for temporal context.

    Not a nn.Module — just a data structure. Attention is in TemporalAttention.
    """

    def __init__(self, buffer_len=3):
        self.buffer_len = buffer_len
        self.buffers = {}  # entity_id -> deque(maxlen=buffer_len)

    def update(self, entity_order, features):
        """Store current window's features (detached CPU tensors)."""
        for i, eid in enumerate(entity_order):
            if eid not in self.buffers:
                self.buffers[eid] = deque(maxlen=self.buffer_len)
            self.buffers[eid].append(features[i].detach().cpu())

    def get_history_tensor(self, entity_order, current_features, device):
        """Build (N, L, M) history tensor for temporal attention.

        For entities with < L history windows, zero-pads from the left.
        The last position is always current_features.
        """
        M = current_features.shape[1]
        L = self.buffer_len
        batch = []
        for i, eid in enumerate(entity_order):
            if eid in self.buffers:
                history = list(self.buffers[eid])
            else:
                history = []
            # Pad with zeros from the left
            while len(history) < L - 1:
                history.insert(0, torch.zeros(M))
            # Append current (not yet stored)
            history.append(current_features[i].detach().cpu())
            # Take last L items
            history = history[-L:]
            batch.append(torch.stack(history))
        return torch.stack(batch).to(device)  # (N, L, M)

    def reset(self):
        self.buffers = {}


class ProjectionHead(nn.Module):
    """Projects backbone features to L2-normalized contrastive space."""

    def __init__(self, input_dim, hidden_dim=None, output_dim=32):
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


class PrototypeModule(nn.Module):
    """EMA class prototypes for reversal-robust classification.

    Maintains per-class centroid in representation space, updated via EMA.
    During normal operation, provides auxiliary classification scores fused
    with MLP logits. During distribution reversal, prototype scores dominate.
    """

    def __init__(self, feat_dim, n_classes=2, beta=0.99):
        super().__init__()
        self.n_classes = n_classes
        self.beta = beta
        self.register_buffer('prototypes', torch.zeros(n_classes, feat_dim))
        self.register_buffer('initialized', torch.zeros(n_classes, dtype=torch.bool))

    def update(self, features, labels, beta_override=None):
        """Update prototypes with EMA. features: (N, D), labels: (N,)."""
        beta = beta_override if beta_override is not None else self.beta
        with torch.no_grad():
            for c in range(self.n_classes):
                mask = labels == c
                if mask.any():
                    class_mean = features[mask].detach().mean(dim=0)
                    if self.initialized[c]:
                        self.prototypes[c] = beta * self.prototypes[c] + (1 - beta) * class_mean
                    else:
                        self.prototypes[c] = class_mean
                        self.initialized[c] = True

    def similarity_scores(self, features):
        """Cosine similarity to each prototype. Returns (N, n_classes)."""
        feat_norm = F.normalize(features, dim=1)
        proto_norm = F.normalize(self.prototypes, dim=1)
        return torch.mm(feat_norm, proto_norm.t())

    def predict_scores(self, features):
        """Return positive-class probability from prototype distances."""
        sim = self.similarity_scores(features)  # (N, 2)
        return torch.softmax(sim, dim=1)[:, 1]  # P(attack)

    def is_ready(self):
        return self.initialized.all().item()

    def reset(self):
        self.prototypes.zero_()
        self.initialized.zero_()


def sup_con_loss(features, labels, temperature=0.1, pc_matrix=None,
                 entity_order=None, pc_entity_map=None):
    """Supervised Contrastive Loss with optional manifold guidance.

    Args:
        features: (N, D) L2-normalized embeddings from projection head
        labels: (N,) class labels
        temperature: scaling temperature
        pc_matrix: (n_pc, n_pc) causal coupling matrix from PCDriftForecaster (optional)
        entity_order: list of entity IDs corresponding to features rows (optional)
        pc_entity_map: dict mapping entity_id -> index in pc_matrix (optional)

    When pc_matrix is provided, positive pairs are weighted by causal coupling
    strength: causally coupled same-class entities are pulled closer.
    """
    device = features.device
    N = features.shape[0]
    if N < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Cosine similarity matrix
    sim = torch.mm(features, features.t()) / temperature  # (N, N)

    # Positive mask: same class, excluding self
    labels_col = labels.view(-1, 1)
    mask_pos = (labels_col == labels_col.t()).float()
    mask_pos.fill_diagonal_(0)

    # If no positive pairs exist, return zero
    n_pos = mask_pos.sum(dim=1)
    if (n_pos > 0).sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Optional: manifold-guided weighting
    if pc_matrix is not None and entity_order is not None and pc_entity_map is not None:
        pc_weights = torch.ones(N, N, device=device)
        for i in range(N):
            eid_i = entity_order[i]
            if eid_i not in pc_entity_map:
                continue
            pi = pc_entity_map[eid_i]
            for j in range(N):
                if i == j:
                    continue
                eid_j = entity_order[j]
                if eid_j not in pc_entity_map:
                    continue
                pj = pc_entity_map[eid_j]
                if pi < pc_matrix.shape[0] and pj < pc_matrix.shape[1]:
                    pc_weights[i, j] = 1.0 + float(pc_matrix[pi, pj])
        mask_pos = mask_pos * pc_weights

    # Numerical stability
    logits_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - logits_max.detach()

    # Denominator: exp(sim) for all pairs except self
    mask_all = torch.ones(N, N, device=device)
    mask_all.fill_diagonal_(0)
    exp_sim = torch.exp(sim) * mask_all
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    # Weighted mean over positives
    n_pos_safe = mask_pos.sum(dim=1).clamp(min=1e-8)
    mean_log_prob = (mask_pos * log_prob).sum(dim=1) / n_pos_safe

    valid = mask_pos.sum(dim=1) > 0
    loss = -mean_log_prob[valid].mean()
    return loss


class ContrastiveContinualMLP(nn.Module):
    """MLP with contrastive learning, temporal attention, and prototypes.

    Replaces ContinualMLP. Architecture:
        TemporalAttention -> Backbone [128, 64] -> Classifier + ProjectionHead + Prototypes
    """

    def __init__(self, input_dim, hidden_dims=(128, 64), n_classes=2,
                 dropout=0.2, proj_dim=32, ewc_lambda=0.5):
        super().__init__()
        self.input_dim = input_dim

        # Temporal attention (applied before backbone)
        self.temporal_attn = TemporalAttention(input_dim)

        # Feature backbone (freezable)
        layers = []
        prev = input_dim
        for hd in hidden_dims:
            layers.extend([
                nn.Linear(prev, hd), nn.ReLU(),
                nn.Dropout(dropout), nn.BatchNorm1d(hd)
            ])
            prev = hd
        self.feature_extractor = nn.Sequential(*layers)
        self.feat_dim = prev  # last hidden dim (64)

        # Classifier head
        self.classifier = nn.Linear(prev, n_classes)

        # Projection head (for contrastive loss)
        self.projection = ProjectionHead(prev, prev, proj_dim)

        # Prototype module
        self.prototype = PrototypeModule(prev, n_classes)

        # EWC state
        self.ewc_lambda = ewc_lambda
        self.fisher_dict = {}
        self.optpar_dict = {}

        # Backbone freeze state
        self._backbone_frozen = False

    def forward(self, x, return_all=False):
        """Forward pass.
        Args:
            x: (N, M) input features
            return_all: if True, return (logits, features, projections)
        Returns:
            logits, features [, projections]
        """
        feat = self.feature_extractor(x)
        logits = self.classifier(feat)
        if return_all:
            proj = self.projection(feat)
            return logits, feat, proj
        return logits, feat

    def forward_temporal(self, current, history_seq, return_all=False):
        """Forward with temporal attention applied first.
        current: (N, M), history_seq: (N, L, M)
        """
        x = self.temporal_attn(current, history_seq)
        return self.forward(x, return_all=return_all)

    def predict_with_prototypes(self, features, logits, proto_weight=0.2):
        """Fuse MLP logits with prototype similarity for final scores."""
        scores_mlp = torch.softmax(logits, dim=1)[:, 1]
        if self.prototype.is_ready():
            scores_proto = self.prototype.predict_scores(features)
            return (1 - proto_weight) * scores_mlp + proto_weight * scores_proto
        return scores_mlp

    def freeze_backbone(self):
        """Freeze feature extractor parameters (stable state)."""
        if not self._backbone_frozen:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            self._backbone_frozen = True

    def unfreeze_backbone(self):
        """Unfreeze feature extractor parameters (pre_drift/drift)."""
        if self._backbone_frozen:
            for param in self.feature_extractor.parameters():
                param.requires_grad = True
            self._backbone_frozen = False

    def compute_ewc_loss(self):
        """EWC regularization loss over all trainable parameters."""
        loss = 0.0
        for n, p in self.named_parameters():
            if n in self.fisher_dict and p.requires_grad:
                loss += (self.fisher_dict[n] * (p - self.optpar_dict[n]) ** 2).sum()
        return self.ewc_lambda * loss

    def update_fisher(self, data_loader, device):
        """Compute Fisher information from replay data."""
        self.eval()
        fisher = {n: torch.zeros_like(p)
                  for n, p in self.named_parameters() if p.requires_grad}
        n_batches = 0
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            self.zero_grad()
            out, _ = self(x)
            loss = nn.CrossEntropyLoss()(out, y)
            loss.backward()
            for n, p in self.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data.clone() ** 2
            n_batches += 1
        if n_batches > 0:
            for n in fisher:
                fisher[n] /= n_batches
        for n in fisher:
            if n in self.fisher_dict:
                self.fisher_dict[n] = 0.5 * self.fisher_dict[n] + 0.5 * fisher[n]
            else:
                self.fisher_dict[n] = fisher[n]
        self.optpar_dict = {n: p.data.clone() for n, p in self.named_parameters()}
