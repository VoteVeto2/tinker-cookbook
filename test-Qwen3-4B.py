"""
This script tests the Qwen3-4B-Instruct-2507 model locally(stored in the `model-Qwen3-4B-Instruct-checkpoint` folder)
by generating responses to a set of test prompts. 
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Base model and LoRA adapter paths
BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
LORA_PATH = "./model-Qwen3-4B-Instuct-checkpoint"

print("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

def generate(prompt: str, max_new_tokens: int = 4096) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.1,
            top_p=0.95,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response

# Test questions
test_prompts = [
    "Find the eigenvalues and eigenvectors of the matrix A = [[1, 2], [2, 3]]."
]

print("\n" + "="*50)
print("Testing fine-tuned Qwen3-4B-Instruct model")
print("="*50 + "\n")

for prompt in test_prompts:
    print(f"Q: {prompt}")
    response = generate(prompt)
    print(f"A: {response}\n")
    print("-"*50 + "\n")