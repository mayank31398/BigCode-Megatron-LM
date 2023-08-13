#!/bin/bash

#SBATCH --exclude=nid006865,nid005613,nid005988
#SBATCH --job-name=33b
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
##SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --partition=standard-g
#SBATCH --time=0-00:30:00
#SBATCH --gpus-per-node=mi250:8
#SBATCH --exclusive=user
#SBATCH --hint=nomultithread
#SBATCH --account=project_462000259
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

LEARNING_RATE=1.5e-4

set -euo pipefail


# Training setup
GPUS_PER_NODE=8
# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=32478
NNODES=$SLURM_NNODES
NODE_RANK=$SLURM_PROCID 
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))



CHECKPOINT_PATH="checkpoints"
DATA_PATH="data/debug_text_document"
TOKENIZER_FILE="tokenizer.json"

MICRO_BATCH_SIZE=2
GLOBAL_BATCH_SIZE=512

export WORLD_SIZE=$((SLURM_GPUS_ON_NODE*SLURM_JOB_NUM_NODES))

GPT_ARGS=" \
--tensor-model-parallel-size 4 \
--pipeline-model-parallel-size 4 \
--sequence-parallel \
--num-layers 40 \
--hidden-size 6144 \
--num-attention-heads 48 \
--attention-head-type multiquery \
--init-method-std 0.01275 \
--seq-length 2048 \
--max-position-embeddings 2048 \
--attention-dropout 0.1 \
--hidden-dropout 0.1 \
--micro-batch-size 4 \
--global-batch-size 32 \
--lr 0.0003 \
--min-lr 0.00003 \
--train-iters 250000 \
--lr-decay-iters 250000 \
--lr-decay-style cosine \
--lr-warmup-iters 2000 \
--weight-decay .1 \
--adam-beta2 .95 \
--clip-grad 1.0 \
--bf16 \
--no-gradient-accumulation-fusion \
--fim-rate 0.5 \
--log-interval 10 \
--save-interval 2500 \
--eval-interval 2500 \
--eval-iters 2 \
--use-distributed-optimizer \
--valid-num-workers 0 \
"

OUTPUT_ARGS=" \
    --log-interval 10 \
    --save-interval 1000 \
    --eval-interval 100 \
    --eval-iters 2 \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --structured-logs \
    --structured-logs-dir $CHECKPOINT_PATH/logs \
    "

CMD="torchrun \
--nproc_per_node $GPUS_PER_NODE \
--nnodes $NNODES \
--master_addr $MASTER_ADDR \
--master_port $MASTER_PORT \
pretrain_gpt.py \
$GPT_ARGS \
$OUTPUT_ARGS \
--data-path $DATA_PATH \
"

# Bind masks from Samuel
c=fe

# Bind mask for one thread per core
BIND_MASK_1="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

# Bind mask for two threads per core
BIND_MASK_2="0x${c}00000000000000${c}000000000000,0x${c}00000000000000${c}00000000000000,0x${c}00000000000000${c}0000,0x${c}00000000000000${c}000000,0x${c}00000000000000${c},0x${c}00000000000000${c}00,0x${c}00000000000000${c}00000000,0x${c}00000000000000${c}0000000000"

BIND_MASK="$BIND_MASK_1"
echo "Using --cpu-bind=mask_cpu:$BIND_MASK"

echo $CMD

echo "START $SLURM_JOBID: $(date)"

srun \
    --label \
    --cpu-bind=mask_cpu:$BIND_MASK \
    $CMD

echo "END $SLURM_JOBID: $(date)"
