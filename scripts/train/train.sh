# MI250x

export NCCL_SOCKET_NTHREADS=2
export NCCL_NSOCKS_PERTHREAD=4
export CUDA_DEVICE_MAX_CONNECTIONS=1

DISTRIBUTED_ARGS="\
--nproc_per_node 2 \
--nnodes 1 \
--node_rank 0 \
"

CHECKPOINT_PATH=checkpoints-absolute  # Adjust: Directory to store the checkpoints
TOKENIZER_PATH=../Granite-Megatron-LM/tokenizers/starcoder/tokenizer.json  # Adjust

GPT_ARGS="\
--tensor-model-parallel-size 2 \
--pipeline-model-parallel-size 1 \
--num-layers 32 \
--hidden-size 3072 \
--num-attention-heads 32 \
--attention-head-type multiquery \
--init-method-std 0.01275 \
--seq-length 1024 \
--max-position-embeddings 1024 \
--attention-dropout 0.1 \
--hidden-dropout 0.1 \
--micro-batch-size 1 \
--global-batch-size 4 \
--lr 0.0003 \
--min-lr 0.00003 \
--train-iters 10 \
--lr-decay-iters 10 \
--lr-decay-style cosine \
--lr-warmup-iters 5 \
--weight-decay .1 \
--adam-beta2 .95 \
--clip-grad 1.0 \
--bf16 \
--fim-rate 0 \
--log-interval 10 \
--save-interval 10 \
--eval-interval 10 \
--eval-iters 2 \
--valid-num-workers 0 \
--structured-logs \
--structured-logs-dir $CHECKPOINT_PATH/logs
"


# use these for wandb
# group.add_argument('--wandb-entity-name', type=str, default=None,
#                         help="Name of wandb entity for reporting")
# group.add_argument('--wandb-project-name', type=str, default=None,
#                     help="Name of wandb project")


torchrun $DISTRIBUTED_ARGS \
    pretrain_gpt.py \
    $GPT_ARGS \
    --tokenizer-type TokenizerFromFileWithFIM \
    --tokenizer-file $TOKENIZER_PATH \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --data-path ../Granite-Megatron-LM/data/dataset=Dockerfile \
    --no-gradient-accumulation-fusion
