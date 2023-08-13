TOKENIZER_PATH=tokenizer.json

# tokenize dataset
mkdir -p data
python3 tools/preprocess_data.py \
    --input data.jsonl \
    --output-prefix data/debug \
    --tokenizer-type TokenizerFromFile \
    --tokenizer-file $TOKENIZER_PATH \
    --dataset-impl mmap \
    --append-eod \
    --workers 1 \
    --chunk-size 1000
