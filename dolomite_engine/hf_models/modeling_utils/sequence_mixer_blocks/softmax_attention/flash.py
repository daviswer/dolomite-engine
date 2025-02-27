import torch
from transformers import DynamicCache
from transformers.modeling_flash_attention_utils import _flash_attention_forward

from .....kernels import wait_for_ACT
from ....enums import AttentionHeadType, PositionEmbeddingType
from ...position_embedding import apply_rotary_pos_emb
from .base import Attention


class FlashAttention2(Attention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: DynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # ==========================================================================================
        # hidden_states -> (batch_size, query_length, num_heads * head_dim)
        # ==========================================================================================

        query, key, value = self._prepare_qkv_for_forward(hidden_states)

        # ==========================================================================================
        # query -> (batch_size, num_heads, query_length, head_dim)
        # key -> (batch_size, num_key_value_heads, query_length, head_dim)
        # value -> (batch_size, num_key_value_heads, query_length, head_dim)
        # ==========================================================================================

        if self.position_embedding_type == PositionEmbeddingType.rope:
            query = apply_rotary_pos_emb(query, rope_cos_sin)
            key = apply_rotary_pos_emb(key, rope_cos_sin)

        if past_key_values is not None:
            key, value = past_key_values.update(key, value, self.layer_idx)

        # ==========================================================================================
        # query -> (batch_size, num_heads, query_length, head_dim)
        # key -> (batch_size, num_key_value_heads, key_length, head_dim)
        # value -> (batch_size, num_key_value_heads, key_length, head_dim)
        # ==========================================================================================

        # TODO avoid this extra transpose
        query = query.transpose(1, 2)
        if self.attention_head_type == AttentionHeadType.mqa:
            key = key.squeeze(1).unsqueeze(2)
            value = value.squeeze(1).unsqueeze(2)
        else:
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)

        # ==========================================================================================
        # query -> (batch_size, query_length, num_heads, head_dim)
        # key -> (batch_size, key_length, num_heads, head_dim)
        # value -> (batch_size, key_length, num_heads, head_dim)
        # ==========================================================================================

        batch_size, query_length = query.shape[:2]

        query = wait_for_ACT(query, wait_in_forward=True, wait_in_backward=False)
        key = wait_for_ACT(key, wait_in_forward=True, wait_in_backward=False)
        value = wait_for_ACT(value, wait_in_forward=True, wait_in_backward=False)

        hidden_states = _flash_attention_forward(
            query_states=query,
            key_states=key,
            value_states=value,
            attention_mask=attention_mask,
            query_length=query_length,
            is_causal=self.causal,
            dropout=self.softmax_dropout_p if self.training else 0,
            softmax_scale=self._get_softmax_scale(),
        )

        del query, key, value

        hidden_states = wait_for_ACT(hidden_states, wait_in_forward=False, wait_in_backward=True)
        hidden_states = hidden_states.view(batch_size, query_length, -1)

        # ==========================================================================================
        # hidden_states -> (batch_size, query_length, num_heads * head_dim)
        # ==========================================================================================

        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states
