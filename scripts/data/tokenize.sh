TOKENIZER_PATH=tokenizer.json

# tokenize dataset
mkdir -p data
python tools/preprocess_data.py \
    --input data_raw/debug/mix_uspto.jsonl \
    --output-prefix data/debug \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-path $TOKENIZER_PATH \
    --dataset-impl mmap \
    --append-eod \
    --workers 1 \
    --chunk-size 1000
