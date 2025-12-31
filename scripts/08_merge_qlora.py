import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = "outputs/qlora_adapter"
OUTPUT_PATH = "outputs/merged_model"

# IMPORTANT:
# Load base model in FULL precision for merge (not 4-bit)
print("Loading base model in FP16 for merge...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="cpu",  # merge safely on CPU
)

print("Loading QLoRA adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)

print("Merging adapter into base model...")
model = model.merge_and_unload()

print("Saving merged model...")
model.save_pretrained(OUTPUT_PATH)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.save_pretrained(OUTPUT_PATH)

print("Merged model saved to:", OUTPUT_PATH)
