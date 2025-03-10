import torch
from torch.distributed._tensor.placement_types import Replicate, Shard
from transformers import DynamicCache
from transformers.modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast

from ....distributed import dtensor_to_tensor, tensor_to_dtensor
from ...loss import add_aux_loss, get_autoregressive_language_modeling_loss, get_aux_loss
from ..dense_TP import CausalLMModelMixin_TP


class CausalLMMoEModelMixin_TP(CausalLMModelMixin_TP):
    def forward(
        self,
        input_ids: torch.Tensor | list[list[int]] | None = None,
        past_key_values: DynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | list[list[int]] | None = None,
        position_ids: torch.Tensor | list[list[int]] | None = None,
        inputs_embeds: torch.Tensor | list[list[float]] | None = None,
        labels: torch.Tensor | list[list[int]] | None = None,
        use_cache: bool | None = None,
        return_dict: bool = True,
        output_parallel_lm_logits: bool = False,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> tuple | MoeCausalLMOutputWithPast:
        assert return_dict

        if self.is_pipeline_parallel_enabled:
            prev_aux_loss = past_key_values
            past_key_values = None

        if not self.is_pipeline_parallel_enabled or self.is_first_stage:
            input_ids, position_ids, token_type_ids, labels, cu_seqlens, max_seqlen = self.prepare_inputs_for_model(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                labels=labels,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                use_cache=use_cache,
            )

        transformer_outputs: MoeModelOutputWithPast = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        if not self.is_pipeline_parallel_enabled or self.is_last_stage:
            lm_logits = self.get_lm_logits(transformer_outputs.last_hidden_state)

            if self.m_width is not None:
                lm_logits = lm_logits / self.m_width

        if not self.is_pipeline_parallel_enabled:
            loss = None
            if labels is not None:
                loss = get_autoregressive_language_modeling_loss(
                    lm_logits=lm_logits,
                    labels=labels,
                    cu_seqlens=cu_seqlens,
                    use_padding_free_transformer=self._use_padding_free_transformer,
                    reduction=reduction,
                )

        if self.is_pipeline_parallel_enabled and not self.is_first_stage:
            add_aux_loss(prev_aux_loss)

        if (not self.is_pipeline_parallel_enabled or self.is_last_stage) and not output_parallel_lm_logits:
            # all gather
            lm_logits = tensor_to_dtensor(lm_logits, device_mesh=self.tp_mesh, current_placement=Shard(-1))
            lm_logits = dtensor_to_tensor(lm_logits, device_mesh=self.tp_mesh, desired_placement=Replicate())

        aux_loss = get_aux_loss()

        if self.is_pipeline_parallel_enabled:
            aux_loss = aux_loss.unsqueeze(0)

            if self.is_last_stage:
                output = (lm_logits, aux_loss)
            else:
                output = (transformer_outputs.last_hidden_state, aux_loss)
        else:
            if loss is not None and aux_loss != 0:
                loss = loss + self.router_aux_loss_coef * aux_loss

            output = MoeCausalLMOutputWithPast(
                loss=loss,
                logits=lm_logits,
                past_key_values=transformer_outputs.past_key_values,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
            )

        return output

    def get_dummy_input_tensor(
        self, micro_batch_size: int, sequence_length: int, intermediate_dtype: torch.dtype
    ) -> tuple[torch.Tensor] | torch.Tensor:
        dummy_input = super().get_dummy_input_tensor(micro_batch_size, sequence_length, intermediate_dtype)

        if not self.is_first_stage:
            aux_loss_dummy = torch.empty(1, device=torch.cuda.current_device(), dtype=intermediate_dtype).squeeze(0)
            dummy_input = (dummy_input, aux_loss_dummy)

        return dummy_input

    def get_dummy_output_tensor(
        self,
        micro_batch_size: int,
        sequence_length: int,
        intermediate_dtype: torch.dtype,
        output_parallel_lm_logits_if_possible: bool,
    ) -> tuple[int]:
        dummy_output = super().get_dummy_output_tensor(
            micro_batch_size, sequence_length, intermediate_dtype, output_parallel_lm_logits_if_possible
        )
        aux_loss_dummy = torch.empty(1, device=torch.cuda.current_device(), dtype=intermediate_dtype)
        return dummy_output, aux_loss_dummy
