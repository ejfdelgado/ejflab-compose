from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# cd /tmp/imageia/processor-pyclient/
# python3 llm_train.py

# Load the pre-trained model and tokenizer
model_name = "/root/.cache/gpt4all/Meta-Llama-3-8B-Instruct.Q4_0.gguf"  # Path to GPT-4All model
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare the dataset
data = [
    {"prompt": "Facts about cats:\n", "completion": "Cats sleep 12-16 hours a day. They have retractable claws."},
    {"prompt": "What is a group of cats called?\n", "completion": "A group of cats is called a clowder."}
]
dataset = load_dataset("json", data_files={"train": data})

def tokenize_function(examples):
    return tokenizer(examples["prompt"] + examples["completion"], truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Set training arguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10,
    save_total_limit=2,
    logging_dir="./logs",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")