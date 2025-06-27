from transformers import PreTrainedModel

from ..decoderonly.decoderonly_architecture import (
    DecoderOnlyAttention,
    DecoderOnlyFlashAttention,
    DecoderOnlyLayer,
    DecoderOnlyModel,
    DecoderOnlyWrapper,
)


class ColQwen2_5_LanguageModelWrapper(DecoderOnlyWrapper):
    def convert_to_rbln_causal_lm(self, causal_lm: PreTrainedModel, max_seq_len: int):
        new_layers = []

        for layer in causal_lm.model.layers:
            if self.attn_impl == "eager":
                new_self_attn = DecoderOnlyAttention(
                    layer.self_attn,
                    self.use_attention_mask,
                    self.use_position_ids,
                    kvcache_block_size=self.kvcache_block_size,
                )
            elif self.attn_impl == "flash_attn":
                new_self_attn = DecoderOnlyFlashAttention(
                    layer.self_attn,
                    kvcache_partition_len=self.kvcache_partition_len,
                    kvcache_block_size=self.kvcache_block_size,
                    use_attention_mask=self.use_attention_mask,
                    use_position_ids=self.use_position_ids,
                )
            else:
                raise NotImplementedError(f"Unknwon attn : {self.attn_impl}")

            new_layer = DecoderOnlyLayer(layer, new_self_attn)
            new_layers.append(new_layer)

        new_model = DecoderOnlyModel(
            causal_lm.model,
            new_layers,
            partition_len=self.kvcache_partition_len,
            max_seq_len=max_seq_len,
            kvcache_block_size=self.kvcache_block_size,
            use_learned_pos_emb=self.use_learned_pos_emb,
            sliding_window_layers=self.sliding_window_layers,
        )

        # custom_text_projection layer from origin model
        self.custom_text_proj = causal_lm.custom_text_proj
        return new_model

    def prepare_forward_args(self, *args):
        args = list(args)
        input_ids = None if self.use_inputs_embeds else args.pop(0)
        inputs_embeds = args.pop(0) if self.use_inputs_embeds else None
        cache_position = args.pop(0)
        global_block_tables = args.pop(0)
        local_block_tables = None
        position_embeds = args.pop(0)
        position_ids = None
        attention_mask = args.pop(0) if self.use_attention_mask else None
        past_key_values = args

        if len(past_key_values) != 2 * self.num_hidden_layers:
            raise ValueError(
                f"Different past_key_values to model's config. {len(past_key_values)} != {2 * self.num_hidden_layers}"
            )

        # [key, value] * n_layer -> ( (key, value) ) * n_layer
        # cache shape : batch, n_heads, 1, max_seq_len, head_dim
        _past_key_values = []
        for i in range(self.config.num_hidden_layers):
            key_states = past_key_values[i * 2]
            value_states = past_key_values[i * 2 + 1]
            past_key_value = [key_states, value_states]
            _past_key_values.append(past_key_value)
        past_key_values = _past_key_values

        return (
            input_ids,
            inputs_embeds,
            cache_position,
            global_block_tables,
            local_block_tables,
            attention_mask,
            position_ids,
            past_key_values,
            position_embeds,
        )

    def forward(self, *args):
        (
            input_ids,
            inputs_embeds,
            cache_position,
            global_block_tables,
            local_block_tables,
            attention_mask,
            position_ids,
            past_key_values,
            rotary_emb,
        ) = self.prepare_forward_args(*args)

        last_hidden_states = self.causal_lm(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            position_ids=position_ids,
            past_key_values=past_key_values,
            rotary_emb=rotary_emb,
            global_block_tables=global_block_tables,
            local_block_tables=local_block_tables,
        )
        proj = self.custom_text_proj(last_hidden_states)

        return proj
