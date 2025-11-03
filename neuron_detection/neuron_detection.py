import collections
import logging
import math
import os
import pickle
import random

import click
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


COMPONENTS = (
    ATTN_Q,
    ATTN_K,
    ATTN_V,
    ATTN_O,
    FFN_UP,
    FFN_GATE,
    FFN_DOWN,
) = (
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_o",
    "ffn_up",
    "ffn_gate",
    "ffn_down",
)


def Prompting(
        model,
        tokenizer,
        prompt,
        components,
        top_number_attn: int = 1000,
        top_number_ffn: int = 2000,
        top_number_layer: int = 24,
        suppress_attention_mask_creation: bool = True,
):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    hidden_states_as_lists = collections.defaultdict(list)
    hidden_scores = collections.defaultdict(dict)
    activate_keys = collections.defaultdict(dict)
    logits_dict = {}
    layer_keys = []


    def remove_attention_mask_pre_hook(_module, _args, kwargs):
        kwargs["attention_mask"] = None


    def mlp_hook_with_layer_idx(layer_idx):

        def mlp_hook(module, args, _output):
            hidden_states = args[0]

            if FFN_UP in components:
                hidden_scores[FFN_UP][layer_idx] = torch.sum(torch.abs(module.up_proj(hidden_states)), dim=1).squeeze().tolist()

            if FFN_DOWN in components:
                hidden_scores[FFN_DOWN][layer_idx] = torch.sum(torch.abs(module.up_proj(hidden_states)), dim=1).squeeze().tolist()
        
        return mlp_hook


    # While `*Attention` classes have a `layer_idx` attribute, we pass the layer index
    # explicitly in case some `*Attention` classes do not have this attribute.
    def self_attn_hook_with_layer_idx(layer_idx):

        # Most of the code is copied from transformers.models.mistral.modeling_mistral.MistralAttention.forward()
        def self_attn_hook(module, _args, kwargs, output):
            nonlocal hidden_scores

            hidden_states = kwargs["hidden_states"]
            attention_mask = kwargs.get("attention_mask")

            orig_attn_output, orig_attn_weights, orig_past_key_value = output

            bsz, q_len, _ = hidden_states.size()

            query_states = module.q_proj(hidden_states)
            query_states = query_states.view(bsz, q_len, module.num_heads, module.head_dim).transpose(1, 2)

            if orig_past_key_value is not None:
                key_states = orig_past_key_value.key_cache[layer_idx]
                value_states = orig_past_key_value.value_cache[layer_idx]

                value_states_from_hidden_states = module.v_proj(hidden_states)
                value_states_from_hidden_states = value_states_from_hidden_states.view(
                    bsz, q_len, module.num_key_value_heads, module.head_dim).transpose(1, 2)

                cos, sin = module.rotary_emb(value_states_from_hidden_states, kwargs.get("position_ids"))
                query_states, _ = _apply_rotary_pos_emb(query_states, None, cos, sin)
            else:
                # There is no cache to update when the original `past_key_value` argument
                # was omitted from `forward()` (i.e. was None).
                key_states = module.k_proj(hidden_states)
                value_states = module.v_proj(hidden_states)

                key_states = key_states.view(bsz, q_len, module.num_key_value_heads, module.head_dim).transpose(1, 2)
                value_states = value_states.view(bsz, q_len, module.num_key_value_heads, module.head_dim).transpose(1, 2)

                cos, sin = module.rotary_emb(value_states, kwargs.get("position_ids"))
                query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin)

            key_states = _repeat_kv(key_states, module.num_key_value_groups)
            value_states = _repeat_kv(value_states, module.num_key_value_groups)

            if hasattr(module, "scaling"):
                scaling = module.scaling
            else:
                scaling = 1 / math.sqrt(module.head_dim)

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
            attn_weights_temp = torch.matmul(query_states.transpose(2, 3).unsqueeze(-1), key_states.transpose(2, 3).unsqueeze(-1).transpose(-2, -1))

            if getattr(module.config, "attn_logit_softcapping", None) is not None:
                attn_weights = attn_weights / module.config.attn_logit_softcapping
                attn_weights = torch.tanh(attn_weights)
                attn_weights = attn_weights * module.config.attn_logit_softcapping

            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask

                attn_weights_temp = attn_weights_temp + attention_mask.unsqueeze(2)

            attn_weights_temp = attn_weights.unsqueeze(2).expand(-1, -1, query_states.size()[-1], -1, -1) - attn_weights_temp
            attn_weights_temp = nn.functional.softmax(attn_weights_temp, dim=-1, dtype=torch.float32).to(query_states.dtype)

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=module.attention_dropout, training=module.training)
            attn_output = torch.matmul(attn_weights, value_states)

            if attn_output.size() != (bsz, module.num_heads, q_len, module.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, module.num_heads, q_len, module.head_dim)}, but is"
                    f" {attn_output.size()}"
                )

            attn_weights_temp = attn_weights_temp - attn_weights.unsqueeze(2).expand(-1, -1, query_states.size()[-1], -1, -1)
            attn_weights_temp = attn_weights_temp ** 2
            attn_weights_temp = attn_weights_temp.sum(dim=(-2, -1)).view(-1)

            attn_weights = None

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output_temp = attn_output.reshape(bsz, q_len, module.hidden_size)

            if module.config.model_type == "llama":
                attn_output = attn_output.reshape(bsz, q_len, module.hidden_size)
            else:
                attn_output = attn_output.view(bsz, q_len, -1)

            attn_output_o = attn_output

            attn_output = module.o_proj(attn_output)

            if ATTN_Q in COMPONENTS:
                hidden_scores[ATTN_Q][layer_idx] = attn_weights_temp.squeeze().tolist()
            if ATTN_K in COMPONENTS:
                hidden_scores[ATTN_K][layer_idx] = attn_weights_temp.squeeze().tolist()
            if ATTN_V in COMPONENTS:
                hidden_scores[ATTN_V][layer_idx] = torch.sum(torch.abs(attn_output_temp), dim=1).squeeze().tolist()
            if ATTN_O in COMPONENTS:
                hidden_scores[ATTN_O][layer_idx] = torch.sum(torch.abs(attn_output_o), dim=1).squeeze().tolist()

            return orig_attn_output, orig_attn_weights, orig_past_key_value

        return self_attn_hook


    def store_activated_neurons_hook(module, _args, output):
        nonlocal hidden_states_as_lists, activate_keys, logits_dict, layer_keys

        summed_data = collections.defaultdict(dict)
        components_for_combined_data = [FFN_UP, ATTN_Q, ATTN_V]

        if all(component in components for component in components_for_combined_data):
            for component_ in components_for_combined_data:
                summed_data[component_] = {key: sum(value) for key, value in hidden_scores[component_].items()}

            combined_data = {
                key: summed_data[FFN_UP][key] * 3 + summed_data[ATTN_Q][key] * 2 + summed_data[ATTN_V][key] * 2
                for key in summed_data[FFN_UP]
            }
        else:
            combined_data = None

        for layer_index in range(len(module.model.layers)):
            logits_dict[layer_index] = module.lm_head(output.hidden_states[layer_index]).float()

            for component in COMPONENTS:
                if component in components:
                    if component.startswith("attn_"):
                        top_number = top_number_attn
                    elif component.startswith("ffn_"):
                        top_number = top_number_ffn
                    else:
                        raise AssertionError(f"component '{component}' not supported")

                    activate_keys[component][layer_index] = np.argsort(hidden_scores[component][layer_index])[-top_number:][::-1]

            if combined_data is not None:
                sorted_items = sorted(combined_data.items(), key=lambda item: item[1])
                layer_keys = [item[0] for item in sorted_items[-top_number_layer:]]
            else:
                layer_keys = []

        # TODO: This may not yield the same results as embedding this code inside `model._sample()` in case of multiple GPUs.
        for layer_index, logit in logits_dict.items():
            hidden_states_as_lists[layer_index].append(torch.argmax(logit, dim=-1))


    mlp_handles = []
    for index, layer in enumerate(model.model.layers):
        mlp_handles.append(
            layer.mlp.register_forward_hook(
                mlp_hook_with_layer_idx(layer_idx=index),
            )
        )
    self_attn_handles = []
    for index, layer in enumerate(model.model.layers):
        self_attn_handles.append(
            layer.self_attn.register_forward_hook(
                self_attn_hook_with_layer_idx(layer_idx=index),
                with_kwargs=True,
            )
        )
    remove_attention_mask_handles = []
    if suppress_attention_mask_creation:
        for index, layer in enumerate(model.model.layers):
            remove_attention_mask_handles.append(
                layer.register_forward_pre_hook(
                    remove_attention_mask_pre_hook,
                    with_kwargs=True,
                )
            )
            remove_attention_mask_handles.append(
                layer.self_attn.register_forward_pre_hook(
                    remove_attention_mask_pre_hook,
                    with_kwargs=True,
                )
            )
        remove_attention_mask_handles.append(model.register_forward_pre_hook(remove_attention_mask_pre_hook, with_kwargs=True))
    store_activated_neurons_handle = model.register_forward_hook(store_activated_neurons_hook)
    try:
        outputs = model.generate(
            input_ids=inputs.input_ids,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )
    finally:
        for handle in mlp_handles:
            handle.remove()
        for handle in self_attn_handles:
            handle.remove()
        for handle in remove_attention_mask_handles:
            handle.remove()
        store_activated_neurons_handle.remove()

    hidden_states = {}
    for layer_index in hidden_states_as_lists:
        hidden_states[layer_index] = torch.cat(hidden_states_as_lists[layer_index], dim=-1)
    
    hidden_embed = {}
    for layer_index in range(len(model.model.layers)):
        hidden_embed[layer_index] = tokenizer.decode(hidden_states[layer_index][0])
    answer = tokenizer.decode(outputs[0][0]).replace('<pad> ', '')
    answer = answer.replace('</s>', '')
    
    return hidden_embed, answer, activate_keys, layer_keys


def _detect_neurons(
        model_name,
        model,
        tokenizer,
        language_corpus,
        corpus_sample_size,
        components,
        top_number_attn,
        top_number_ffn,
        suppress_attention_mask_creation,
):
    file_path = os.path.join("corpus_all", language_corpus + ".txt")
    with open(file_path, 'r') as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines]
    lines = random.sample(lines, corpus_sample_size)

    activate_keys_list = collections.defaultdict(list)

    error_count = 0

    for prompt in tqdm(lines):
        try:
            _hidden_embed, _answer, activate_keys, _layer_keys = Prompting(
                model,
                tokenizer,
                prompt,
                components,
                top_number_attn=top_number_attn,
                top_number_ffn=top_number_ffn,
                suppress_attention_mask_creation=suppress_attention_mask_creation,
            )
        except torch.cuda.OutOfMemoryError as e:
            error_count += 1
            print(f"Encountered the following error: {e}")
        else:
            for component in activate_keys:
                activate_keys_list[component].append(activate_keys[component])

    common_elements = {}
    for component in activate_keys_list:
        common_elements[component] = _get_common_elements_for_component(activate_keys_list, component)

    file_path = os.path.join(
        "output_neurons",
        *model_name.split('/'),
        language_corpus,
        f"detected_neurons_{top_number_attn}_{top_number_ffn}_{corpus_sample_size - error_count}.txt",
    )

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'wb') as file:
        pickle.dump(common_elements, file)


def _get_common_elements_for_component(activate_keys_list, component):
    common_elements = {}

    for key in activate_keys_list[component][0]:
        if all(key in d for d in activate_keys_list[component]):
            # Extract corresponding arrays and find common elements
            arrays = [d[key] for d in activate_keys_list[component]]
            common_elements[key] = set.intersection(*map(set, arrays))

    return common_elements


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
# and modified to allow ignoring processing q or k
def _apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor` or `None`): The query tensor, or `None` if processing this tensor is not needed.
        k (`torch.Tensor` or `None`): The key tensor, or `None` if processing this tensor is not needed.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    if q is not None:
        q_embed = (q * cos) + (_rotate_half(q) * sin)
    else:
        q_embed = None
    
    if k is not None:
        k_embed = (k * cos) + (_rotate_half(k) * sin)
    else:
        k_embed = None

    return q_embed, k_embed


# Copied from transformers.models.llama.modeling_llama.rotate_half
def _rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


@click.command()
@click.option(
    "--language-corpus",
    "-c",
    "language_corpora",
    default=("english",),
    multiple=True,
    help="Language(s) for which to load a corpus.",
)
@click.option(
    "--corpus-sample-size",
    default=1000,
    help="Number of documents to sample from the corpus given by --language-corpus.",
)
@click.option(
    "--model-name",
    default="mistralai/Mistral-7B-Instruct-v0.2",
    help="Name of the model for which to detect language-specific neurons.",
)
@click.option(
    "--component",
    "-o",
    "components",
    default=(
        ATTN_Q,
        ATTN_K,
        ATTN_V,
        FFN_UP,
        FFN_DOWN,
    ),
    type=click.Choice(COMPONENTS),
    multiple=True,
    help="Component(s) to detect neurons in.",
)
@click.option(
    "--top-number-attn",
    default=1000,
    help="Number of neurons with the highest activation values to consider on attention layers.",
)
@click.option(
    "--top-number-ffn",
    default=2000,
    help="Number of neurons with the highest activation values to consider on feedforward layers.",
)
@click.option(
    "--attn-implementation",
    default="eager",
    help=(
        "Attention implementation to use for neuron detection. "
        "For the list of possible values, see: "
        "https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.from_pretrained.attn_implementation"
    ),
)
@click.option(
    "--suppress-attention-mask/--no-suppress-attention-mask",
    default=True,
    help=(
        "Suppresses/allows automatic creation of attention masks for consistency with the original repository."
    ),
)
@click.option(
    "--random-seed",
    default=112,
    help=(
        "Fixed random seed for sampling from the corpus given by --language-corpus."
        " If you do not wish to use a fixed random seed, use -1."
    ),
)
def main(
    language_corpora,
    corpus_sample_size,
    model_name,
    components,
    top_number_attn,
    top_number_ffn,
    attn_implementation,
    suppress_attention_mask,
    random_seed,
):
    """Detects language-specific neurons based on the method by Zhao et al.: https://arxiv.org/abs/2402.18815
    
    Original repository: https://github.com/DAMO-NLP-SG/multilingual_analysis
    """

    logging.basicConfig(level=logging.INFO)

    if random_seed != -1:
        random.seed(random_seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        attn_implementation=attn_implementation,
    )

    # Force greedy search
    model.generation_config.do_sample = False
    model.generation_config.num_beams = 1

    for language_corpus in language_corpora:
        _detect_neurons(
            model_name,
            model,
            tokenizer,
            language_corpus,
            corpus_sample_size,
            components,
            top_number_attn,
            top_number_ffn,
            suppress_attention_mask,
        )


if __name__ == "__main__":
    main()
