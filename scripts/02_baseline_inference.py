import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_PATH = "data/raw/final_dataset_instruct.jsonl"
OUTPUT_PATH = "outputs/baseline_outputs.txt"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto"
)

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
        out.write("=== MODEL OUTPUT ===\n")
        out.write(answer + "\n")

print("Baseline inference completed.")
print(f"Results saved to {OUTPUT_PATH}")
