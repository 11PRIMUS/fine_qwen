import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType

model_n="Qwen/Qwen1.5-0.5B"
data_path="emotion1.jsonl"
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
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=True
)

model=AutoModelForCausalLM.from_pretrained(
    model_n,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
lora_config=LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj","v_proj"],
    lora_dropout=0.1,
    bias="none",
    tasktype=TaskType.CAUSAL_LM
)
model=get_peft_model(model,lora_config)

args=TrainingArguments( #training
    output_dir="emotion2",
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
