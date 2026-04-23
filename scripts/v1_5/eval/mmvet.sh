#!/bin/bash

CKPT="llava-v1.5-divt0.65-7b"
THRESHOLD=0.65

python -m llava.eval.model_vqa \
    --model-path $CKPT \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/${CKPT}.jsonl \
    --temperature 0 \
    --threshold $THRESHOLD \
    --conv-mode vicuna_v1

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/${CKPT}.jsonl \
    --dst ./playground/data/eval/mm-vet/results/${CKPT}.json

