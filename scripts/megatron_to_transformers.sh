INPUT_PATH=checkpoints
OUTPUT_PATH=unsharded

python tools/checkpoint_util.py \
    --model-type GPT  \
    --load-dir $INPUT_PATH \
    --save-dir $OUTPUT_PATH \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --use-distributed-optimizer

export PYTHONPATH=./

INPUT_PATH=$OUTPUT_PATH/iter_0020000
OUTPUT_PATH=transformers_compatible/iter_0020000
TOKENIZER_PATH=../Granite-Megatron-LM/tokenizers/starcoder/tokenizer.json

python tools/megatron_to_huggingface.py \
    --path_to_checkpoint $INPUT_PATH \
    --save_dir $OUTPUT_PATH \
    --tokenizer-file $TOKENIZER_PATH \
    --tokenizer-type TokenizerFromFileWithFIM \
    --custom_model \
    --safetensors

# clean unsharded checkpoints once done
