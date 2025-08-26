import json
import torch
import numpy as np
import wandb
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    RobertaForSequenceClassification,
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, hamming_loss
import argparse
import os

toxic_masculinity_indicators = [
    'is_anti_feminism',
    'is_anti_lgbt',
    'is_emotional_repression',
    'is_endorsement_of_aggression',
    'is_female_malice_stereotype',
    'is_female_objectification',
    'is_gender_rigidity',
    'is_male_separatism',
    'is_male_marginalization',
    'is_masculine_superiority',
    'is_sexual_dominance',
    'is_intersectional',
    'is_alt_right_terminology'
]

class MultilabelDataset:
    def __init__(self, jsonl_path, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data(jsonl_path)
    
    def load_data(self, jsonl_path):
        data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                data.append(item)
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(item['labels'], dtype=torch.float)
        }

def create_hf_dataset(jsonl_path, tokenizer, max_length=256):
    """Create HuggingFace Dataset from JSONL file"""
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)
    
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        tokenized['labels'] = examples['labels']
        return tokenized
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list(data)
    dataset = dataset.map(tokenize_function, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    return dataset

def compute_metrics(eval_pred):
    """Compute metrics for multilabel classification"""
    predictions = eval_pred.predictions
    labels = eval_pred.labels
    
    # Apply sigmoid to get probabilities
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # Convert to binary predictions (threshold = 0.5)
    y_pred = (probs > 0.5).int().numpy()
    y_true = labels.astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_macro = recall_score(y_true, y_pred, average='macro')

    return {
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro
    }

class MultilabelTrainer(Trainer):
    """Custom trainer for multilabel classification with BCEWithLogitsLoss"""
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        # Use BCEWithLogitsLoss for multilabel classification
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)
        
        return (loss, outputs) if return_outputs else loss

def main():
    parser = argparse.ArgumentParser(description='Multilabel Hate Speech Classification Training')
    parser.add_argument('--train_data', type=str, required=True, help='Path to training JSONL file')
    parser.add_argument('--val_data', type=str, required=True, help='Path to validation JSONL file')
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory')
    parser.add_argument('--model_name', type=str, default='roberta-base', help='Base model name')
    parser.add_argument('--max_length', type=int, default=256, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=32, help='Evaluation batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--warmup_steps', type=int, default=500, help='Number of warmup steps')
    parser.add_argument('--logging_steps', type=int, default=50, help='Logging frequency')
    parser.add_argument('--save_steps', type=int, default=500, help='Save frequency')
    parser.add_argument('--eval_steps', type=int, default=500, help='Evaluation frequency')
    parser.add_argument('--wandb_api_key', type=str, default='YOUR_WANDB_API_KEY_HERE', help='Weights & Biases API key')
    parser.add_argument('--project_name', type=str, default='hate-speech-classification', help='W&B project name')
    parser.add_argument('--run_name', type=str, default='roberta-lora-multilabel', help='W&B run name')
    parser.add_argument('--num_labels', type=int, default=13, help='Number of labels for multilabel classification')
    
    # LoRA parameters
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout')
    
    args = parser.parse_args()
    
    # Set up wandb
    if args.wandb_api_key != 'YOUR_WANDB_API_KEY_HERE':
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
        wandb.init(project=args.project_name, name=args.run_name)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = create_hf_dataset(args.train_data, tokenizer, args.max_length)
    val_dataset = create_hf_dataset(args.val_data, tokenizer, args.max_length)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Load model
    print("Loading model...")
    model = RobertaForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=args.num_labels,
        problem_type="multi_label_classification"
    )
    
    # Configure LoRA
    print("Setting up LoRA...")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["query", "key", "value", "dense"],
        modules_to_save=["classifier"]
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        logging_dir=f'{args.output_dir}/logs',
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        learning_rate=args.learning_rate,
        optim="adamw_torch",
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to="wandb" if args.wandb_api_key != 'YOUR_WANDB_API_KEY_HERE' else None,
        run_name=args.run_name,
        seed=42,
        fp16=torch.cuda.is_available(),
        label_names=toxic_masculinity_indicators
    )
    
    # Initialize trainer
    trainer = MultilabelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train the model
    # print("Starting training...")
    # trainer.train()
    
    # # Save the final model
    # print("Saving model...")
    # trainer.save_model(f"{args.output_dir}/final_model")
    # tokenizer.save_pretrained(f"{args.output_dir}/final_model")
    
    # # Final evaluation
    # print("Final evaluation...")
    # eval_results = trainer.evaluate()
    # print("Final evaluation results:")
    # for key, value in eval_results.items():
    #     print(f"{key}: {value}")
    
    # # Save evaluation results
    # with open(f"{args.output_dir}/eval_results.json", "w") as f:
    #     json.dump(eval_results, f, indent=2)
    
    # print("Training completed!")

if __name__ == "__main__":
    main()