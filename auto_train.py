import torch
from dataets import load_dataset
from transformers import Autokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling

from peft import get_peft_model, LoraConfig, TaskType

model_n="Qwen/Qwen1.5-0.5B"
data=""
dataset=load_dataset("json",data_f=data,split="train")

def f_prompt(example):
    return{
        "text":"f"### Emotion:{exmaple['emotion]}\n###n Input:{example["input"]}\n Response:{example['response]}""
    }
dataset=dataset.map(f_prompt)
tokenizer=Autokenizer.from_pretrained(model_n,trust_remote_code=True)
tokenizer.pad_token=tokenizer.eos_token

def tokenize(example):
    return tokenizer(example["text"],truncation=True,padding="max_length",max_length=512)

tokenized=dataset.map(tokenize,batched=True)

model=AutoModelForCausalLM.from_pretrained(
    model_n,load_in_8bit=True,device_map="auto",trust_remote_code=True
)
lora_config=LoraConfig(
    r=8,loara_alpha=32,target_modules=["q_proj","v_proj"],
    lora_dropout=0.1,bias="none",tasktype=TaskType.CAUSAL_LM
)
model=get_perft_model(model,lora_config)

