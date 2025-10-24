import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

# ---------------- CONFIG ----------------
MODEL_NAME = "meta-llama/Llama-7b-hf"  # or Llama-13b if memory allows
CORPUS_FILE = "dapt_data_final.txt"          # Your domain text file
SAVE_DIR = "./dapt_llama_model"
SEQ_LENGTH = 512                        # Max token length per sample
BATCH_SIZE = 1                           # Adjust per GPU memory
EPOCHS = 1                              # Start small
GRAD_ACCUM = 4                          # Simulate larger batch
FP16 = True                             # Mixed precision training
GPU_ID = 0                               # The MIG GPU assigned
# ----------------------------------------

os.makedirs(SAVE_DIR, exist_ok=True)

# Set GPU device explicitly
device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
print(f"Using device: {device}")

# ---------------- LOAD TOKENIZER & MODEL ----------------
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if FP16 else torch.float32,
).to(device)

# ---------------- LOAD DATASET ----------------
print("Loading dataset...")
dataset = load_dataset("text", data_files={"train": CORPUS_FILE})

def tokenize_fn(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=SEQ_LENGTH,
        padding="max_length",
    )

tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

# ---------------- DATA COLLATOR ----------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # causal LM
)

# ---------------- TRAINING ARGUMENTS ----------------
training_args = TrainingArguments(
    output_dir=SAVE_DIR,
    overwrite_output_dir=True,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    save_strategy="epoch",
    logging_steps=50,
    learning_rate=2e-5,
    fp16=FP16,
    report_to="none",  # disable wandb/tensorboard
    save_total_limit=2,
)

# ---------------- TRAINER ----------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
)

# ---------------- TRAIN ----------------
print("Starting DAPT training on single GPU...")
trainer.train()

# ---------------- SAVE MODEL ----------------
print(f"Saving trained model to {SAVE_DIR}...")
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print("DAPT training complete!")
