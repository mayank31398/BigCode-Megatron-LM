####################################################################################################

# Copyright (c) 2021-, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

####################################################################################################

#
# Note: If when running this conversion script you're getting an exception:
#     ModuleNotFoundError: No module named 'megatron.model.enums'
# you need to tell python where to find the clone of Megatron-LM, e.g.:
#
# cd /tmp
# git clone https://github.com/NVIDIA/Megatron-LM
# PYTHONPATH=/tmp/Megatron-LM python src/transformers/models/megatron_gpt2/convert_megatron_gpt2_checkpoint.py ...
#
# if you already have it cloned elsewhere, simply adjust the path to the existing path
#
# If the training was done using a Megatron-LM fork, e.g.,
# https://github.com/microsoft/Megatron-DeepSpeed/ then chances are that you need to have that one
# in your path, i.e., /path/to/Megatron-DeepSpeed/
#

import os
import re
import sys
from argparse import ArgumentParser, Namespace

import torch

# custom version of transformers has GPTMegatron classes and is only needed when exporting a custom model
from transformers import AutoModelForCausalLM, GPTBigCodeConfig


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from megatron.tokenizer import build_tokenizer


def sort_dict_by_keys(x: dict) -> dict:
    keys = list(x.keys())
    keys.sort()

    result = {}
    for k in keys:
        result[k] = x[k]

    return result


def recursive_print(name, val, spaces=0):
    # Format the message.
    if name is None:
        msg = None
    else:
        fmt = "." * max(0, spaces - 2) + "# {:" + str(50 - spaces) + "s}"
        msg = fmt.format(name)

    # Print and recurse (if needed).
    if isinstance(val, dict):
        if msg is not None:
            print(msg)
        for k in val.keys():
            recursive_print(k, val[k], spaces + 2)
    elif isinstance(val, torch.Tensor):
        print(msg, ":", val.size())
    else:
        print(msg, ":", val)


# The simple map of names for "automated" rules.
NAME_MAP = {
    "attention.dense": ".attn.c_proj.",
    "self_attention.dense": ".attn.c_proj.",
    "mlp.dense_h_to_4h": ".mlp.c_fc.",
    "mlp.dense_4h_to_h": ".mlp.c_proj.",
    "self_attention.query_key_value": ".attn.c_attn.",
    "self_attention.query": ".attn.q_attn.",
    "self_attention.key_value": ".attn.kv_attn.",
}


def get_args() -> Namespace:
    # Create the argument parser.
    parser = ArgumentParser()

    group = parser.add_argument_group("checkpoint")
    group.add_argument("--print-checkpoint-structure", action="store_true")
    group.add_argument(
        "--path_to_checkpoint",
        type=str,
        help="Path to the checkpoint file (.zip archive or direct .pt file)",
    )
    group.add_argument(
        "--custom_model",
        action="store_true",
        help="Save as custom model so it can be used with huggingface transformers.",
    )
    group.add_argument(
        "--save_dir", help="Path where the converted model is saved. Will use the checkpoint directory if not provided"
    )
    group.add_argument("--safetensors", action="store_true", help="save in safetensors format")

    group = parser.add_argument_group(title="tokenizer")
    group.add_argument("--make-vocab-size-divisible-by", type=int, default=128, help="make vocab size divisible by")
    group.add_argument(
        "--tokenizer-type",
        type=str,
        required=True,
        choices=[
            "BertWordPieceLowerCase",
            "BertWordPieceCase",
            "GPT2BPETokenizer",
            "GPT2BPETokenizerWithFIM",
            "HuggingFaceTokenizer",
            "TokenizerFromFile",
            "TokenizerFromFileWithFIM",
        ],
        help="What type of tokenizer to use.",
    )
    group.add_argument("--vocab-file", type=str, default=None, help="Path to the vocab file")
    group.add_argument("--merge-file", type=str, default=None, help="Path to the BPE merge file (if necessary).")
    group.add_argument("--tokenizer-file", type=str, default=None, help="Path to the tokenizer")

    args = parser.parse_args()
    return args


def make_vocab_size_divisible_by(vocab_size: int, divisor: int) -> int:
    n = vocab_size // divisor
    if n * divisor != vocab_size:
        n += 1
    return n * divisor


def convert_megatron_checkpoint(
    input_state_dict: dict,
    padded_vocab_size: int,
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
    custom_model: bool,
    checkpoint_args: Namespace,
):
    # The converted output model.
    output_state_dict = {}

    if checkpoint_args.attention_head_type in ["multihead", "groupedquery"]:
        multi_query = False
    else:
        assert checkpoint_args.attention_head_type == "multiquery"
        multi_query = True

    attention_softmax_in_fp32 = (
        checkpoint_args.attention_softmax_in_fp32 or checkpoint_args.apply_query_key_layer_scaling
    )

    config = GPTBigCodeConfig(
        vocab_size=padded_vocab_size,
        n_positions=checkpoint_args.max_position_embeddings,
        n_embd=checkpoint_args.hidden_size,
        n_layer=checkpoint_args.num_layers,
        n_head=checkpoint_args.num_attention_heads,
        n_inner=checkpoint_args.ffn_hidden_size,
        activation_function="pytorch_gelu_tanh" if checkpoint_args.bias_gelu_fusion else "gelu",
        multi_query=multi_query,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        scale_attn_weights=True,
        use_cache=True,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        attention_softmax_in_fp32=attention_softmax_in_fp32,
        scale_attention_softmax_in_fp32=True,
    )

    # The model.
    model = input_state_dict["model"]["language_model"]

    # The word embeddings, truncated to to vocab_size rows.
    word_embeddings = model["embedding"]["word_embeddings"]["weight"][: config.vocab_size, :]
    output_state_dict["transformer.wte.weight"] = word_embeddings

    # The position embeddings.
    output_state_dict["transformer.wpe.weight"] = model["embedding"]["position_embeddings"]["weight"]
    model_max_length = output_state_dict["transformer.wpe.weight"].shape[0]

    # The transformer.
    transformer = model["transformer"] if "transformer" in model else model["encoder"]

    # The regex to extract layer names.
    layer_re = re.compile("layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")

    # Extract the layers.
    for key, val in transformer.items():
        # Match the name.
        m = layer_re.match(key)

        # Stop if that's not a layer
        if m is None:
            break

        # The index of the layer.
        layer_idx = int(m.group(1))
        # The name of the operation.
        op_name = m.group(2)
        # Is it a weight or a bias?
        weight_or_bias = m.group(3)

        # The name of the layer.
        layer_name = f"transformer.h.{layer_idx}"

        # For layernorm(s), simply store the layer norm.
        if op_name.endswith("layernorm"):
            ln_name = "ln_1" if op_name.startswith("input") else "ln_2"
            output_state_dict[layer_name + "." + ln_name + "." + weight_or_bias] = val

        # Concatenate QKV matrix.
        elif op_name == "self_attention.key_value":
            # Query is before key_value in the dict.
            query = output_state_dict.pop(layer_name + ".attn.q_attn." + weight_or_bias)
            out_val = torch.cat([query, val], dim=0)
            output_state_dict[layer_name + ".attn.c_attn." + weight_or_bias] = out_val

        # Copy the parameters.
        else:
            output_state_dict[layer_name + NAME_MAP[op_name] + weight_or_bias] = val

    # DEBUG.
    assert config.n_layer == layer_idx + 1

    # The final layernorm.
    output_state_dict["transformer.ln_f.weight"] = transformer["final_layernorm.weight"]
    output_state_dict["transformer.ln_f.bias"] = transformer["final_layernorm.bias"]

    # For LM head, transformers' wants the matrix to weight embeddings.
    output_state_dict["lm_head.weight"] = word_embeddings

    # It should be done!
    return config, output_state_dict, model_max_length


####################################################################################################


def main() -> None:
    args = get_args()

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.tensor_model_parallel_size = 1

    tokenizer = build_tokenizer(args)

    print(f"Extracting PyTorch state dictionary from {args.path_to_checkpoint}")

    input_state_dict = torch.load(
        os.path.join(args.path_to_checkpoint, "mp_rank_00", "model_optim_rng.pt"), map_location="cpu"
    )

    checkpoint_args = input_state_dict["args"]

    # Convert.
    print("Converting")

    config, output_state_dict, model_max_length = convert_megatron_checkpoint(
        input_state_dict,
        args.padded_vocab_size,
        tokenizer.eod,
        tokenizer.eod,
        tokenizer.eod if tokenizer.pad is None else tokenizer.pad,
        args.custom_model,
        checkpoint_args,
    )

    # Print the structure of converted state dict.
    if args.print_checkpoint_structure:
        recursive_print(None, output_state_dict)

    tokenizer.tokenizer.model_max_length = model_max_length
    tokenizer.tokenizer.save_pretrained(args.save_dir, legacy_format=False)

    hf_model = AutoModelForCausalLM.from_config(config)

    hf_model.load_state_dict(output_state_dict)
    hf_model.save_pretrained(args.save_dir, safe_serialization=args.safetensors)


if __name__ == "__main__":
    main()
