import logging

import torch
from peft import LoraConfig

from my_lora import loadModelForLoRA, chatAssistantOnlyMapper, paddingDatasetBatcher, T_PEFT_TYPE

logging.basicConfig(level=logging.INFO)
print('Started "test_qlora.py"')

MODEL_ID = 'Qwen/Qwen3.5-2B' # 'andreasmartin/apertus-v1.5-8b-text' 'Qwen/Qwen3.5-4B' 'Qwen/Qwen3.5-0.8B'
OUTPUT_DIR = './artifacts/lora'
PEFT_TYPE: T_PEFT_TYPE = 'qlora4'
DEVICE = torch.device('cuda') # torch.device('cpu') torch.device('tpu')

BATCH_SIZE = 2
MAX_STEPS = 30
ACCUM_STEPS = 4
LEARNING_RATE = 2e-4

# Loading model
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.0,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
tokenizer, model = loadModelForLoRA(model_id=MODEL_ID, device=DEVICE, peft_type=PEFT_TYPE, lora_config=lora_config)

# Training data
examples = [
    ("What is the capital of France?", "Paris."),
    ("What is the capital of Italy?", "Rome."),
    ("What is the capital of Germany?", "Berlin."),
    ("What is the capital of Spain?", "Madrid."),
]

# Training encoder
mapper = chatAssistantOnlyMapper(tokenizer, device=DEVICE, chat_template_kwargs={ 'enable_thinking': False })
def encode_example(question: str, answer: str):
    return mapper(
        [ { 'role': 'user', 'content': question }, ],
        answer,
    )

dataset = [
    encode_example(question, answer)
    for question, answer in examples
]

# Optimizer
optimizer = torch.optim.AdamW(
    (p for p in model.parameters() if p.requires_grad),
    lr=LEARNING_RATE,
)

# PyTorch training
model.train()
optimizer.zero_grad(set_to_none=True)

data_iterator = paddingDatasetBatcher(tokenizer, dataset, batch_size=BATCH_SIZE, cycle=True)

for step in range(1, MAX_STEPS + 1):
    total_loss = 0.0

    for _ in range(ACCUM_STEPS):
        batch = next(data_iterator)

        outputs = model(**batch)
        loss = outputs.loss

        (loss / ACCUM_STEPS).backward()
        total_loss += loss.detach().item()

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    print(
        f"Step {step}/{MAX_STEPS} "
        f"loss={total_loss / ACCUM_STEPS:.4f}",
        flush=True,
    )

# ---------------------------
# Save adapter only
# ---------------------------
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Saved LoRA adapter to {OUTPUT_DIR}")
