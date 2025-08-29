import json
import torch
import torch.nn as nn
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback,
    AutoModelForSequenceClassification
)
from peft import AutoPeftModelForSequenceClassification, PeftConfig, PeftModel
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

def create_hf_dataset(jsonl_path, tokenizer, max_length=256):
    """Create HuggingFace Dataset from JSONL file"""
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            # Convert to numpy array with float32 dtype
            item['labels'] = np.array(item['labels'], dtype=np.float32).tolist()
            data.append(item)
    
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=max_length
        )
        # Convert all labels to float32
        tokenized['labels'] = [np.array(labels, dtype=np.float32).tolist() 
                              for labels in examples['labels']]
        return tokenized
    
    dataset = Dataset.from_list(data)
    dataset = dataset.map(tokenize_function, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    return dataset

def compute_metrics(eval_pred):
    """Compute metrics for multilabel classification"""
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids
    
    # Apply sigmoid to get probabilities
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # Convert to binary predictions (threshold = 0.5)
    y_pred = (probs > 0.5).int().numpy()
    y_true = labels.astype(int)
    
    # Calculate metrics with zero_division parameter to handle edge cases
    accuracy = accuracy_score(y_true, y_pred)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)

    res = {
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro
    }
    return res

class MultilabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch = None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Instantiate BCE with Logits Loss
        loss_fct = nn.BCEWithLogitsLoss()

        # Compute the loss
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), 
                        labels.float().view(-1, self.model.config.num_labels))

        return (loss, outputs) if return_outputs else loss

def main():
    parser = argparse.ArgumentParser(description='Multilabel Hate Speech Classification Evaluation')
    parser.add_argument('--test_data', type=str, required=True, help='Path to test JSONL file')
    parser.add_argument('--load_model_path', type=str, required=True, help='Path to a saved PEFT model directory')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory for evaluation results')
    parser.add_argument('--max_length', type=int, default=256, help='Maximum sequence length')
    parser.add_argument('--eval_batch_size', type=int, default=32, help='Evaluation batch size')
    parser.add_argument('--num_labels', type=int, default=13, help='Number of labels for multilabel classification')
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.load_model_path)
    
    # Load datasets
    print("Loading datasets...")
    test_dataset = create_hf_dataset(args.test_data, tokenizer, args.max_length)
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Load model
    print(f"Loading PEFT model from {args.load_model_path}...")
    model = AutoPeftModelForSequenceClassification.from_pretrained(
        args.load_model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        num_labels=args.num_labels
    ).to(device)

    # fail safe? in case previous does't work
    # config = PeftConfig.from_pretrained(args.load_model_path)
    # base_model = AutoModelForSequenceClassification.from_pretrained(
    #     config.base_model_name_or_path,
    #     num_labels=args.num_labels,
    #     problem_type="multi_label_classification"
    # )
    # model = PeftModel.from_pretrained(base_model, args.load_model_path)
    
    model = model.to(device)
    model.eval()

   #  eval_args = TrainingArguments(
   #      output_dir=args.output_dir,
   #      per_device_eval_batch_size=args.eval_batch_size,
   #      report_to=None, # Disable wandb for a simple evaluation script
   #      dataloader_pin_memory=False,
   #      remove_unused_columns=False,
   #  )
    
   # trainer = MultilabelTrainer(
   #      model=model,
   #      args=eval_args,
   #      eval_dataset=test_dataset,
   #      tokenizer=tokenizer,
   #      compute_metrics=compute_metrics,
   #      callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
   #  )

   #  results = trainer.evaluate()
    print("model loaded!")

if __name__ == "__main__":
    main()