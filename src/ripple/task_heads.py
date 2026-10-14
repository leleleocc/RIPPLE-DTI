from typing import Dict, Optional

import torch
from torch import Tensor, nn


class PredictionMLP(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.network(inputs)


class RIPPLETaskHeads(nn.Module):
    """Task readouts over the pocket-conditioned interaction state."""

    def __init__(
        self,
        dim: int = 128,
        num_affinity_endpoints: int = 4,
        num_effect_classes: int = 3,
        num_mechanism_classes: int = 6,
        dropout: float = 0.1,
        local_gate_bias: float = -2.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_affinity_endpoints = num_affinity_endpoints
        self.global_projection = nn.Sequential(
            nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim)
        )
        self.local_projection = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.local_gate = nn.Linear(dim * 2, dim)
        nn.init.constant_(self.local_gate.bias, local_gate_bias)
        self.binding_norm = nn.LayerNorm(dim)

        self.endpoint_embedding = nn.Embedding(num_affinity_endpoints + 1, dim)
        self.interaction_head = PredictionMLP(dim, dim, 1, dropout)
        self.affinity_head = PredictionMLP(dim * 2, dim, 1, dropout)
        effect_input_dim = dim * 4
        self.effect_head = PredictionMLP(
            effect_input_dim, dim * 2, num_effect_classes, dropout
        )
        self.mechanism_head = PredictionMLP(
            effect_input_dim, dim * 2, num_mechanism_classes, dropout
        )
        self.allosteric_pocket_head = PredictionMLP(dim, dim, 1, dropout)

    def forward(
        self,
        global_context: Tensor,
        pair_context: Tensor,
        function_state: Tensor,
        pocket_global: Tensor,
        pocket_valid_mask: Tensor,
        endpoint_type: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        batch_size = pair_context.size(0)
        if global_context.shape != (batch_size, self.dim * 2):
            raise ValueError("global_context must have shape [B, 2D]")
        global_state = self.global_projection(global_context)
        local_state = self.local_projection(pair_context)
        local_gate = torch.sigmoid(
            self.local_gate(torch.cat([global_state, local_state], dim=-1))
        )
        binding_state = self.binding_norm(global_state + local_gate * local_state)

        if endpoint_type is None:
            endpoint_index = torch.zeros(
                batch_size, device=binding_state.device, dtype=torch.long
            )
        else:
            endpoint_index = (
                endpoint_type.to(device=binding_state.device, dtype=torch.long) + 1
            )
            endpoint_index = endpoint_index.clamp(
                min=0, max=self.num_affinity_endpoints
            )
        endpoint_state = self.endpoint_embedding(endpoint_index)

        effect_features = torch.cat(
            [
                pair_context,
                function_state,
                pair_context * function_state,
                torch.abs(pair_context - function_state),
            ],
            dim=-1,
        )
        allosteric_pocket_logit = self.allosteric_pocket_head(pocket_global).squeeze(-1)
        allosteric_pocket_logit = allosteric_pocket_logit.masked_fill(
            ~pocket_valid_mask,
            torch.finfo(allosteric_pocket_logit.dtype).min,
        )

        return {
            "interaction_logit": self.interaction_head(binding_state).squeeze(-1),
            "affinity": self.affinity_head(
                torch.cat([binding_state, endpoint_state], dim=-1)
            ).squeeze(-1),
            "effect_logits": self.effect_head(effect_features),
            "mechanism_logits": self.mechanism_head(effect_features),
            "allosteric_pocket_logits": allosteric_pocket_logit,
            "binding_state": binding_state,
            "local_fusion_gate": local_gate,
        }
