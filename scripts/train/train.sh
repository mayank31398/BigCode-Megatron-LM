# MI250x

export NCCL_SOCKET_NTHREADS=2
export NCCL_NSOCKS_PERTHREAD=4
export CUDA_DEVICE_MAX_CONNECTIONS=1

DISTRIBUTED_ARGS="\
--nproc_per_node $GPUS_PER_NODE \
--nnodes $NNODES \
--node_rank $NODE_RANK \
--master_addr $MASTER_ADDR \
--master_port $MASTER_PORT \
"

CHECKPOINT_PATH=checkpoints-absolute  # Adjust: Directory to store the checkpoints
TOKENIZER_PATH=tokenizer.json  # Adjust

GPT_ARGS="\
--tensor-model-parallel-size 2 \
--pipeline-model-parallel-size 1 \
--num-layers 32 \
--hidden-size 3072 \
--num-attention-heads 32 \
--attention-head-type multiquery \
--init-method-std 0.01275 \
--seq-length 2048 \
--max-position-embeddings 2048 \
--attention-dropout 0.1 \
--hidden-dropout 0.1 \
--micro-batch-size 8 \
--global-batch-size 2048 \
--lr 0.0003 \
--min-lr 0.00003 \
--train-iters 300000 \
--lr-decay-iters 300000 \
--lr-decay-style cosine \
--lr-warmup-iters 2000 \
--weight-decay .1 \
--adam-beta2 .95 \
--clip-grad 1.0 \
--bf16 \
--use-flash-attn \
--fim-rate 0 \
--log-interval 10 \
--save-interval 2500 \
--eval-interval 2500 \
--eval-iters 2 \
--use-distributed-optimizer \
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
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-path $TOKENIZER_PATH \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --data-path data/debug_text_document \
    --fix-infiniband
