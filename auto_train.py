import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
    
model_n="sshleifer/tiny-gpt2"
data_path="em10000.jsonl"
dataset=load_dataset("json",data_files=data_path,split="train")

def f_prompt(example):
    return{
        "text":f"### Emotion:{example['emotion']}\n### Input:{example['input']}\n### Response:{example['response']}"
    }
dataset=dataset.map(f_prompt)

tokenizer=AutoTokenizer.from_pretrained(model_n,trust_remote_code=True)
tokenizer.pad_token=tokenizer.eos_token

def tokenize(example):
    return tokenizer(example["text"],truncation=True,padding="max_length",max_length=512)

tokenized=dataset.map(tokenize,batched=True)

model=AutoModelForCausalLM.from_pretrained(
    model_n,
    device_map="auto",
    trust_remote_code=True
)
lora_config=LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["c_attn", "c_proj"], #
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model=get_peft_model(model,lora_config)

args=TrainingArguments( #training
    output_dir="emotion",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_dir="logs",
    logging_steps=5,
    save_strategy="epoch",
)
trainer=Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer,mlm=False),

)
trainer.train()
model.save_pretrained("emotion")
tokenizer.save_pretrained("emotion")
