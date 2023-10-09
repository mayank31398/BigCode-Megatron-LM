import os
from argparse import Namespace

import torch
from transformers import AutoConfig, set_seed, AutoModelForCausalLM

from megatron.arguments import parse_args, validate_args
from megatron.checkpointing import load_args_from_checkpoint, load_checkpoint
from megatron.core import mpu
from megatron.initialize import set_global_variables
from megatron.utils import get_ltor_masks_and_position_ids
from pretrain_gpt import model_provider


UNSHARDED_CHECKPOINT_PATH = "unsharded"
HF_CHECKPOINT_PATH = "transformers_compatible"
ITERATION = 20000

DTYPE = torch.float32
FLASH_ATTENTION = False

SEED = 42

set_seed(SEED)


def get_num_params(model: torch.nn.Module) -> int:
    return sum([param.numel() for param in model.parameters()])


class MegatronModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        megatron_model = model_provider()
        load_checkpoint([megatron_model], None, None, iteration=ITERATION)
        self.model = megatron_model

        self.model = self.model.to(DTYPE)
        self.vocab_size = self.model.language_model.embedding.word_embeddings.num_embeddings

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        attention_mask, _, position_ids = get_ltor_masks_and_position_ids(input_ids, -1, False, False, False)
        return self.model(input_ids, position_ids, attention_mask)

    def vocab_forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        _, _, position_ids = get_ltor_masks_and_position_ids(input_ids, -1, False, False, False)
        return self.model.language_model.embedding(input_ids, position_ids).transpose(0, 1)

    def layer_forward(
        self,
        hidden_states: torch.Tensor,
        causal_mask: torch.Tensor,
        alibi_bias: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        # TODO causal_mask is not handled correctly for MHA
        causal_mask = ~causal_mask

        return self.model.language_model.encoder.layers[layer_idx](
            hidden_states.transpose(0, 1), causal_mask.transpose(1, 2), rotary_pos_emb=rotary_pos_emb
        ).transpose(0, 1)

    def get_alibi(self, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model.language_model.encoder.layers[0]._build_alibi_tensor(2048, 32, 8).cuda()


class HFModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        model_name = os.path.join(HF_CHECKPOINT_PATH, "iter_{:07d}".format(ITERATION))

        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=DTYPE).to("cuda:1")

        self.vocab_size = self.config.vocab_size
        self.num_attention_heads = self.config.num_attention_heads
        self.hidden_size = self.config.hidden_size

        self.multi_query = self.config.multi_query

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, attention_mask=attention_mask)

    def vocab_forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        return self.model.transformer._get_initial_hidden_state(input_ids, None, position_ids, None)

    def layer_forward(
        self,
        hidden_states: torch.Tensor,
        causal_mask: torch.Tensor,
        alibi_bias: torch.Tensor,
        rope_cos_sin: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        hidden_states = self.model.transformer.h[layer_idx](
            hidden_states=hidden_states,
            attention_mask=causal_mask,
        )
        return hidden_states[0]

    def get_alibi(self, attention_mask: torch.Tensor) -> torch.Tensor:
        # we pass query_length = 1 to avoid repeat and easy comparison
        return self.model.transformer._get_alibi_bias(
            attention_mask, attention_mask.shape[0], 1, attention_mask.shape[1], attention_mask.device, DTYPE
        )

    def get_position_embedding_type(self) -> str:
        return self.model.transformer.position_embedding_type.value


class ModelComparator:
    def __init__(self) -> None:
        self.megatron_model = MegatronModel().to("cuda:0")
        self.hf_model = HFModel()

        print(f"{'-' * 25} megatron model {'-' * 25}")
        print(self.megatron_model)
        print(f"{'-' * 25} HFcompat model {'-' * 25}")
        print(self.hf_model)
        print("-" * 66)

        self.megatron_model.eval()
        self.hf_model.eval()

        self.vocab_size = min(self.megatron_model.vocab_size, self.hf_model.vocab_size)
        self.num_attention_heads = self.hf_model.num_attention_heads
        self.hidden_size = self.hf_model.hidden_size

        self.multi_query = self.hf_model.multi_query

    @classmethod
    def print_error_percentage_bins(cls, a: torch.Tensor, b: torch.Tensor, tolerance: float) -> None:
        percent_elements_with_tolerable_error = (torch.abs(a - b) <= tolerance).sum() / a.numel() * 100
        percent_elements_with_tolerable_error = "{:.2f}".format(percent_elements_with_tolerable_error)
        print(f"{percent_elements_with_tolerable_error} % elements fall within tolerance = {tolerance}")

    @classmethod
    def are_equal_tensors(self, a: torch.Tensor, b: torch.Tensor, eps: float = 0, verbose: bool = False) -> bool:
        if verbose:
            print(a.shape, b.shape)
            print(a)
            print(b)
            print(torch.abs(a - b))
            print(torch.abs(a - b).max())
            self.print_error_percentage_bins(a, b, 0.02)
            self.print_error_percentage_bins(a, b, 0.015)
            self.print_error_percentage_bins(a, b, 0.01)
            self.print_error_percentage_bins(a, b, 0.005)
            self.print_error_percentage_bins(a, b, 0.002)
            self.print_error_percentage_bins(a, b, 0.001)
            self.print_error_percentage_bins(a, b, 0.0005)
            self.print_error_percentage_bins(a, b, 0.0002)
        return (torch.abs(a - b) <= eps).all()

    def _compare_vocab_forward(self) -> None:
        input_ids, _, position_ids, _, _, _, _ = self._get_dummy_inputs()

        assert self.are_equal_tensors(
            self.hf_model.vocab_forward(input_ids, position_ids)[..., : self.vocab_size],
            self.megatron_model.vocab_forward(input_ids, position_ids)[..., : self.vocab_size],
            verbose=True,
        )

    def _compare_forward(self) -> None:
        input_ids, attention_mask, _, _, _, _, _ = self._get_dummy_inputs()

        assert self.are_equal_tensors(
            self.hf_model(input_ids.to("cuda:1"), attention_mask.to("cuda:1")).logits[..., : self.vocab_size],
            self.megatron_model(input_ids, attention_mask)[..., : self.vocab_size].to("cuda:1"),
            1e-2,
            True,
        )

    def _compare_layer_forward(self) -> None:
        _, _, _, hidden_states, causal_mask, alibi_bias, rotary_pos_emb = self._get_dummy_inputs()

        megatron_output = self.megatron_model.layer_forward(hidden_states, causal_mask, alibi_bias, rotary_pos_emb, 7)
        hf_output = self.hf_model.layer_forward(hidden_states.to("cuda:1"), causal_mask.to("cuda:1"), alibi_bias, rotary_pos_emb, 7)

        assert self.are_equal_tensors(hf_output, megatron_output.to("cuda:1"), 1e-2, True)

    def _compare_alibi(self) -> None:
        if self.hf_model.get_position_embedding_type() != "alibi":
            return

        megatron_alibi = self.megatron_model.get_alibi(None)

        # micro_batch_size
        batch_size = megatron_alibi.shape[0] // self.num_attention_heads
        attention_mask = torch.ones(batch_size, megatron_alibi.shape[2]).cuda()

        hf_alibi = self.hf_model.get_alibi(attention_mask)
        megatron_alibi = megatron_alibi.reshape(hf_alibi.shape)

        assert self.are_equal_tensors(hf_alibi, megatron_alibi, 1e-3)

    def _get_dummy_inputs(self) -> torch.Tensor:
        input_ids = torch.tensor([[3, 4, 5, 6], [7, 8, 9, 10]]).cuda()
        attention_mask = torch.ones(2, 4, dtype=torch.long).cuda()
        hidden_states = torch.randn(2, 4, self.hidden_size).to(DTYPE).cuda()
        position_ids = (attention_mask.cumsum(dim=-1) - 1).masked_fill_(attention_mask == 0, 0)

        causal_mask = torch.tril(torch.ones((2, 4, 4), dtype=torch.bool, device="cuda")).unsqueeze(2)
        rotary_pos_emb = None
        alibi_bias = None

        return input_ids, attention_mask, position_ids, hidden_states, causal_mask, alibi_bias, rotary_pos_emb

    @torch.inference_mode()
    def compare(self) -> None:
        # self._compare_alibi()
        # self._compare_vocab_forward()
        # self._compare_layer_forward()
        self._compare_forward()


def get_args() -> Namespace:
    args = parse_args()

    args.load = UNSHARDED_CHECKPOINT_PATH
    args.iteration = ITERATION
    args.masked_softmax_fusion = False
    args.masked_softmax_fusion = False
    args.bias_gelu_fusion = False
    args.bias_dropout_fusion = False
    args.async_tensor_model_parallel_allreduce = False
    args.use_cpu_initialization = True
    args.micro_batch_size = 2
    args.seq_length = 4
    args.no_load_optim = True
    args.no_save_optim = True
    args.no_load_rng = True
    args.no_save_rng = True
    args.no_initialization = True
    args.model_type = "GPT"

    if DTYPE == torch.float32:
        args.bf16 = False
        args.fp16 = False
    elif DTYPE == torch.float16:
        args.bf16 = False
        args.fp16 = True
    elif DTYPE == torch.bfloat16:
        args.bf16 = True
        args.fp16 = False

    args.params_dtype = DTYPE
    args.use_flash_attn = FLASH_ATTENTION

    if args.use_flash_attn:
        assert args.params_dtype in [torch.float16, torch.bfloat16]

    args, checkpoint_args = load_args_from_checkpoint(args)

    # These are arguments that we are either changing, or cause problems for validation if they are set
    # Note that some of these deal with T5 so will need to be changed if we support T5.
    args_to_keep = [
        "async_tensor_model_parallel_allreduce",
        "bf16",
        "finetune",
        "bias_dropout_fusion",
        "bias_gelu_fusion",
        "distribute_saved_activations",
        "encoder_num_layers",
        "encoder_seq_length",
        "end_weight_decay",
        "load",
        "lr_decay_iters",
        "lr_warmup_fraction",
        "lr_warmup_iters",
        "masked_softmax_fusion",
        "micro_batch_size",
        "no_load_optim",
        "no_load_rng",
        "no_save_optim",
        "no_save_rng",
        "num_layers_per_virtual_pipeline_stage",
        "params_dtype",
        "params_dtype",
        "perform_initialization",
        "pipeline_model_parallel_size",
        "recompute_granularity",
        "save",
        "save_interval",
        "save_sharded_optim",
        "seq_length",
        "sequence_parallel",
        "start_weight_decay",
        "tensor_model_parallel_size",
        "tokenizer_model",
        "train_iters",
        "use_cpu_initialization",
        "use_flash_attn",
        "virtual_pipeline_model_parallel_size",
        "vocab_file",
        "world_size",
    ]

    for arg, value in vars(checkpoint_args).items():
        if arg in args_to_keep:
            continue
        if not hasattr(args, arg):
            print(f"Checkpoint had argument {arg} but new arguments does not have this.")
            continue
        if getattr(args, arg) != value:
            print(f"Overwriting default {arg} value {getattr(args, arg)} with value from checkpoint {value}.")
            setattr(args, arg, value)

    validate_args(args)

    return args


def main() -> None:
    args = get_args()

    mpu.set_tensor_model_parallel_world_size(1)
    mpu.set_pipeline_model_parallel_world_size(1)
    mpu.set_tensor_model_parallel_rank(0)
    mpu.set_pipeline_model_parallel_rank(0)
    mpu._set_global_memory_buffer()

    set_global_variables(args, False)

    model_comparator = ModelComparator()
    model_comparator.compare()


if __name__ == "__main__":
    main()
