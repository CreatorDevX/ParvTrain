from typing import Optional

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from config import ParvHFConfig
from model import ParvModel


class ParvForCausalLM(PreTrainedModel):
    config_class = ParvHFConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def __init__(self, config: ParvHFConfig):
        super().__init__(config)
        self.model = ParvModel(config.to_model_config())

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> CausalLMOutputWithPast:
        out = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_kv_cache=use_cache,
            update_aux_loss=labels is not None,
        )
        logits = out["logits"]
        aux_loss = out["aux_loss"]
        hidden = out["hidden_states"]

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            loss = ce_loss + aux_loss

        if output_hidden_states:
            return CausalLMOutputWithPast(
                loss=loss, logits=logits, hidden_states=hidden, past_key_values=None,
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=None)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        return {"input_ids": input_ids, "use_cache": True}

    def _reorder_cache(self, past_key_values, beam_idx):
        return past_key_values
