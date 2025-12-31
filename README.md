# AI-ML-SaaS-LLM-Fine-Tune-For-Android

## End-to-End RAG Distillation → LoRA / QLoRA → GGUF Pipeline for Android & Edge

---

## 1. Project Purpose

This repository implements a **fully local, reproducible fine-tuning pipeline**
to convert a **RAG-distilled instruction dataset** into a **mobile-ready LLM**
using:

- LoRA and QLoRA fine-tuning
- Adapter merging
- GGUF conversion
- INT4 quantization for edge / Android deployment

This repository **does NOT perform RAG distillation**.  
It **starts from an already distilled dataset**.

---

## 2. Architectural Separation (IMPORTANT)

This project is designed to work **together with** a separate repository:

AI-ML-SaaS-RAG-Distil → produces final_dataset_instruct.jsonl
AI-ML-SaaS-LLM-Fine-Tune-For-Android (this repo) → fine-tunes & deploys


Pipeline boundary:

[RAG Teacher Repo]
↓
final_dataset_instruct.jsonl
↓
[THIS Fine-Tuning Repo]


Scripts `01–03` are **post-RAG utilities**, not RAG logic.

---

## 3. High-Level Pipeline


Distilled Instruction Dataset
↓
Dataset Validation
↓
Baseline Inference (Base Model)
↓
Subset Preparation (optional)
↓
LoRA Training (validation)
↓
QLoRA Training (low-VRAM)
↓
Adapter Merge
↓
HF → GGUF Conversion
↓
INT4 Quantization (Q4_K_M)
↓
Android / Edge Inference (llama.cpp)


---

## 4. Repository Structure (EXACT)

AI-ML-Fine-Tune/
│
├── data/
│ └── raw/
│ └── final_dataset_instruct.jsonl
│
├── scripts/
│ ├── 01_validate_dataset.py
│ ├── 02_baseline_inference.py
│ ├── 03_prepare_subset.py
│ ├── 04_train_lora.py
│ ├── 05_infer_with_lora.py
│ ├── 06_qlo_ra_preflight.py
│ ├── 07_train_qlora.py
│ └── 08_merge_qlora.py
│
├── outputs/ (ignored by git)
│ ├── lora_adapter/
│ ├── qlora_adapter/
│ ├── merged_model/
│ ├── tinyllama-qlora-fp16.gguf
│ └── tinyllama-qlora-q4_k_m.gguf
│
├── .gitignore
└── README.md



---

## 5. Script Responsibilities (CLEAR)

### 01_validate_dataset.py
Validates the distilled dataset:
- JSONL integrity
- required fields
- empty values
- schema safety

Purpose: **fail fast before training**

---

### 02_baseline_inference.py
Runs inference on the **base model before fine-tuning**.

Purpose:
- capture baseline behavior
- compare with LoRA / QLoRA results

---

### 03_prepare_subset.py
Creates a small dataset subset for:
- fast experiments
- low-risk pipeline testing
- avoiding long training runs

Purpose: **development utility**

---

### 04_train_lora.py
LoRA fine-tuning on the base model.

Purpose:
- verify training logic
- confirm adapter wiring
- validate dataset usability

---

### 05_infer_with_lora.py
Inference using LoRA-trained adapters.

Purpose:
- compare against baseline
- confirm learning signal

---

### 06_qlo_ra_preflight.py
Loads the base model in **4-bit NF4**.

Purpose:
- verify bitsandbytes
- confirm GPU compatibility
- preflight before QLoRA

---

### 07_train_qlora.py
QLoRA training:
- base model frozen in 4-bit
- adapters trained in FP16

Purpose:
- low-VRAM fine-tuning
- edge-ready learning

---

### 08_merge_qlora.py
Merges QLoRA adapters into the base model.

Purpose:
- produce a **single standalone HF model**
- required before GGUF conversion

---

## 6. Model & Training Configuration

- Base model: **TinyLLaMA-1.1B**
- Training method: **QLoRA**
- GPU used: GTX 1650 (4 GB)
- Precision:
  - Base model: 4-bit NF4 (training)
  - Adapters: FP16
- Runtime target: **CPU-only**

---

## 7. GGUF Conversion & Quantization

After adapter merge:

1. Convert Hugging Face model → GGUF (FP16)
2. Quantize GGUF → **Q4_K_M**

Final artifact:

tinyllama-qlora-q4_k_m.gguf


Characteristics:
- INT4
- ~500–550 MB
- Fully offline
- No Python / PyTorch at runtime

---

## 8. Deployment Target

- Android / Edge devices
- Inference via `llama.cpp`
- CPU-only execution
- Suitable for offline knowledge-vault apps
  (e.g., InfinoVault / EdgeRAG)

---

## 9. Scope & Intent

This repository focuses on:

- pipeline correctness
- reproducibility
- edge deployability

It does **not** focus on:
- dataset scale
- benchmark accuracy
- general-purpose chat quality

Those are future extensions.

---

## 10. Key Guarantees

- No cloud APIs
- No vendor lock-in
- Fully local training
- Mobile-deployable artifact
- Clean separation from RAG logic

---

## 11. Author

**Kazi Fahim Lateef**  
Senior Software Engineer  
Focus: RAG systems, edge LLMs, applied ML

Project context: **InfinoVault / EdgeRAG**

