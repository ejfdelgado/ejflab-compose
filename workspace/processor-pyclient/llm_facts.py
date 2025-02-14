from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from huggingface_hub import login
import os

# cd /tmp/imageia/processor-pyclient/
# python3 llm_facts.py
login(token=os.environ['HUGGING_FACE_TOKEN'])

# Step 1: Load the Dataset
# Load the facts dataset from a text file
data_file = "./facts.txt"  # Replace with the path to your file
dataset = load_dataset("text", data_files={"train": data_file})

# Step 2: Initialize Tokenizer and Model
#model_name = "/root/.cache/gpt4all/Meta-Llama-3-8B-Instruct.Q4_0.gguf"
#model_name = "meta-llama/Llama-3.1-8B-Instruct"
model_name = "NousResearch/Hermes-3-Llama-3.2-3B"
#model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


# Step 3: Tokenize the Dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Step 4: Set Training Arguments
training_args = TrainingArguments(
    output_dir="./fact_model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=100,
    evaluation_strategy="epoch",# no epoch
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=100,
)

# Step 5: Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    tokenizer=tokenizer,
)

# Step 6: Train the Model
trainer.train()

# Step 7: Save the Fine-Tuned Model
trainer.save_model("./fact_model")
tokenizer.save_pretrained("./fact_model")

print("Training complete. The model is saved in ./fact_model.")