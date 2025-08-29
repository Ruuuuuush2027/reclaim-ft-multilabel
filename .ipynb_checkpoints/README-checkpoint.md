# reclaim-ft-multilabel

To train, run
```
python train.py \
    --train_data "train.jsonl" \
    --val_data "eval.jsonl" \
    --output_dir "./results" \
    --model_name "roberta-base" \
    --max_length 256 \
    --batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 1e-4 \
    --num_epochs 5 \
    --warmup_steps 500 \
    --logging_steps 50 \
    --save_steps 500 \
    --eval_steps 500 \
    --wandb_api_key "YOUR_ACTUAL_WANDB_API_KEY" \
    --project_name "hate-speech-classification" \
    --run_name "roberta-lora-multilabel-v1" \
    --num_labels 13 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05
```

dataset location: https://github.com/Will-Dolan/ort-frames