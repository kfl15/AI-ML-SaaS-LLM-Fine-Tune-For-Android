import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_PATH = "outputs/lora_adapter"
DATA_PATH = "data/raw/final_dataset_instruct.jsonl"
OUTPUT_PATH = "outputs/lora_outputs.txt"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

questions = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        obj = json.loads(line)
        questions.append(obj["instruction"])

with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
    for i, q in enumerate(questions, 1):
        prompt = q.strip() + "\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False
            )

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        out.write(f"\n=== QUESTION {i} ===\n")
        out.write(prompt + "\n")
        out.write("=== LORA OUTPUT ===\n")
        out.write(answer + "\n")

print("LoRA inference completed.")
print(f"Results saved to {OUTPUT_PATH}")
