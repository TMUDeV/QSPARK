import os, torch
from unsloth import FastLanguageModel, PatchDPOTrainer, is_bfloat16_supported
from datasets import load_dataset
from trl import ORPOTrainer, ORPOConfig

MODEL_NAME = "Qwen/Qwen2.5-Coder-32B-Instruct"
DATA_FILE = "./Formatted_ORPO_Dataset.csv"
OUTPUT_DIR = "./orpo_outputs"
BATCH_SIZE = 2
GRAD_ACCUM = 4
EPOCHS = 2
LORA_RANK = 16
LR = 2e-5

compute_dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16
attn_impl = "flash_attention_2" if torch.cuda.is_bf16_supported() else "sdpa"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=4096,
    dtype=compute_dtype,
    load_in_4bit=True,   
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=[
        "q_proj","k_proj","v_proj","o_proj",
        "gate_proj","up_proj","down_proj",
    ],
    lora_alpha=LORA_RANK,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth", 
    random_state=42,
)

ds = load_dataset("csv", data_files=DATA_FILE)

EOS_TOKEN = tokenizer.eos_token
template = """Below is an instruction that describes a task. Write a response that completes it.

### Instruction:
{}

### Response:
{}"""

def format_sample(example):
    example["prompt"]   = template.format(example["prompt"], "")
    example["chosen"]   = example["chosen"] + EOS_TOKEN
    example["rejected"] = example["rejected"] + EOS_TOKEN
    return example

dataset = ds["train"].map(format_sample)

PatchDPOTrainer()  

trainer = ORPOTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=ORPOConfig(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        logging_steps=5,
        save_strategy="epoch",
        max_length=4096,
        max_prompt_length=2048,
        max_completion_length=2048,
        beta=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        report_to="wandb", 
        output_dir=OUTPUT_DIR,
    ),
)

trainer.train()

FastLanguageModel.for_inference(model)
model.save_pretrained(os.path.join(OUTPUT_DIR, "lora_model"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "lora_model"))
print(f"[DONE] Trained model saved to {OUTPUT_DIR}/lora_model")