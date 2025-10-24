import os
import torch
from datasets import load_dataset
from transformers import LlamaTokenizer, LlamaForCausalLM, Trainer, TrainingArguments
from accelerate import Accelerator
from torch.utils.data import DataLoader

# ---------- CONFIGURATION ----------
# Set the GPUs you want to use (e.g., GPU 0 and 1)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Adjust as per your server's available GPUs

# Path to your domain-specific .txt file
corpus_path = "path/to/your/dapt_corpus.txt"

# Hyperparameters
batch_size = 8
max_length = 1024  # Max token length per sequence
learning_rate = 5e-5
num_train_epochs = 3
logging_dir = "./logs"
output_dir = "./output"
# -------------------------------

# Initialize Accelerator
accelerator = Accelerator()

# Load the tokenizer and model
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-7B-hf")
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-7B-hf")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_length)

# Load and prepare the dataset
dataset = load_dataset('text', data_files={'train': corpus_path})
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Create DataLoader
train_dataset = tokenized_datasets["train"]
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=1,
    evaluation_strategy="epoch",
    logging_dir=logging_dir,
    logging_steps=500,
    save_steps=500,
    save_total_limit=2,
    learning_rate=learning_rate,
    fp16=True,  # Use mixed precision
    report_to="tensorboard",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

# Start training
trainer.train()
