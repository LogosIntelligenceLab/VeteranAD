import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple


class SftLayer(nn.Module):
    def __init__(self,
                 d_edge: int = 128,
                 d_model: int = 128,
                 d_ffn: int = 512,
                 n_head: int = 8,
                 dropout: float = 0.2,
                 update_edge: bool = True) -> None:
        super(SftLayer, self).__init__()
        self.update_edge = update_edge

        self.proj_memory = nn.Sequential(
            nn.Linear(d_model + d_model + d_edge, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        if self.update_edge:
            self.proj_edge = nn.Sequential(
                nn.Linear(d_model, d_edge),
                nn.LayerNorm(d_edge),
                nn.GELU(),
            )
            self.norm_edge = nn.LayerNorm(d_edge)

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_head, dropout=dropout, batch_first=True)

        # Feedforward model
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                edge: Tensor,
                edge_mask: Optional[Tensor] = None) -> Tensor:
        '''
            input:
                query:      (B, Q, d_model)
                key:        (B, K, d_model)
                value:      (B, K, d_model)
                edge:       (B, Q, K, d_edge)
                edge_mask:  (B, Q, K)
        '''
        # update node
        x, edge, memory = self._build_memory(query, key, edge)
        x_prime, _ = self._mha_block(query, memory, attn_mask=None, key_padding_mask=edge_mask)
        x = self.norm2(query + x_prime)
        x = self.norm3(x + self._ff_block(x))
        return x, edge, None

    def _build_memory(self,
                      query: Tensor,
                      key: Tensor,
                      edge: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        '''
            input:
                query:  (B, Q, d_model)
                key:    (B, K, d_model)
                edge:   (B, Q, K, d_edge)
            output:
                :param  (B, Q, K, d_model)
                :param  (B, Q, K, d_edge)
                :param  (B, Q, K, d_model)
        '''
        B, Q, _ = query.shape
        _, K, _ = key.shape

        # 1. build memory
        src_x = query.unsqueeze(dim=2).repeat(1, 1, K, 1)  # (B, Q, K, d_model)
        tar_x = key.unsqueeze(dim=1).repeat(1, Q, 1, 1)    # (B, Q, K, d_model)
        memory = self.proj_memory(torch.cat([edge, src_x, tar_x], dim=-1))  # (B, Q, K, d_model)
        
        # 2. (optional) update edge (with residual)
        if self.update_edge:
            edge = self.norm_edge(edge + self.proj_edge(memory))  # (B, Q, K, d_edge)

        return query, edge, memory

    def _mha_block(self,
                   x: Tensor,
                   mem: Tensor,
                   attn_mask: Optional[Tensor],
                   key_padding_mask: Optional[Tensor]) -> Tensor:
        '''
            input:
                x:                  (B, Q, d_model)
                mem:                (B, Q, K, d_model)
                attn_mask:          (B, Q, K)
                key_padding_mask:   (B, Q, K)
            output:
                :param      (B, Q, d_model)
                :param      (B, Q, K)
        '''
        B, Q, d_model = x.shape
        _, _, K, _ = mem.shape
        x = x.view(B * Q, -1, d_model)
        mem = mem.view(B * Q, K, d_model)
        if attn_mask is not None:
            attn_mask = attn_mask.view(B * Q, K)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.view(B * Q, K)

        x, _ = self.multihead_attn(x, mem, mem,
                                   attn_mask=None,
                                   key_padding_mask=None,
                                   need_weights=False)  # return average attention weights

        x = x.view(B, Q, -1, d_model).squeeze(2)
        return self.dropout2(x), None

    def _ff_block(self,
                  x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class SymmetricFusionTransformer(nn.Module):
    def __init__(self,
                 d_model: int = 128,
                 d_edge: int = 128,
                 n_head: int = 8,
                 n_layer: int = 3,
                 dropout: float = 0.2,
                 update_edge: bool = True):
        super(SymmetricFusionTransformer, self).__init__()

        fusion = []
        for i in range(n_layer):
            need_update_edge = False if i == n_layer - 1 else update_edge
            fusion.append(SftLayer(d_edge=d_edge,
                                   d_model=d_model,
                                   d_ffn=d_model * 2,
                                   n_head=n_head,
                                   dropout=dropout,
                                   update_edge=need_update_edge))
        self.fusion = nn.ModuleList(fusion)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, edge: Tensor, edge_mask: Tensor = None) -> Tensor:
        '''
            query:      (B, Q, d_model)
            key:        (B, K, d_model)
            value:      (B, K, d_model)
            edge:       (B, Q, K, d_edge)
            edge_mask:  (B, Q, K)
        '''
        for mod in self.fusion:
            query, edge, _ = mod(query, key, value, edge, edge_mask)
        return query
