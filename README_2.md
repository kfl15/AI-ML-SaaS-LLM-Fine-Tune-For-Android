```markdown
# README_2 — How to Use This Repository (Step-by-Step Guide)

## Purpose of This Document

This file explains **exactly**:
- what to clone
- in which order
- which script to run after which
- what each phase expects and produces

This is an **execution guide**, not a theory document.

---

## 1. Required Repositories (CLONE BOTH)

This project is designed to work with **two separate repositories**.

### Repository A — RAG Distillation (Teacher)
```

[https://github.com/kfl15/AI-ML-SaaS-RAG-Distil.git](https://github.com/kfl15/AI-ML-SaaS-RAG-Distil.git)

```

### Repository B — Fine-Tuning & Deployment (This Repo)
```

[https://github.com/kfl15/AI-ML-SaaS-LLM-Fine-Tune-For-Android.git](https://github.com/kfl15/AI-ML-SaaS-LLM-Fine-Tune-For-Android.git)

```

### Recommended Folder Layout

```

AI-ML-SaaS/
├── AI-ML-SaaS-RAG-Distil/
└── AI-ML-SaaS-LLM-Fine-Tune-For-Android/

```

The repositories must **not** be merged into one.

---

## 2. Phase 1 — RAG Distillation (Teacher Phase)

📁 Repository:
```

AI-ML-SaaS-RAG-Distil

```

### Goal
Convert raw documents into a **high-quality instruction dataset**.

### Steps (Run in Order)

1. Place documents into:
```

raw_docs/

```

2. Run scripts sequentially:

| Order | Script | What It Does |
|------|-------|-------------|
| 1 | `1_generate_questions.py` | Generates factual questions from documents |
| 2 | `2_run_rag_answers.py` | Answers questions using RAG |
| 3 | `3_filter_samples.py` | Removes weak / hallucinated answers |
| 4 | `4_format_dataset.py` | Converts to instruction format |

### Output of Phase 1

```

final_dataset_instruct.jsonl

```

This file is the **only artifact** required for Phase 2.

---

## 3. Phase 2 — Fine-Tuning & Deployment (This Repository)

📁 Repository:
```

AI-ML-SaaS-LLM-Fine-Tune-For-Android

```

### Dataset Placement

Copy the dataset produced in Phase 1 to:

```

data/raw/final_dataset_instruct.jsonl

```

---

## 4. Phase 2A — Dataset & Baseline Utilities (Optional)

These scripts are **optional but recommended**.
They do **not train** any model.

| Order | Script | Purpose |
|------|-------|--------|
| 01 | `01_validate_dataset.py` | Validates dataset schema and content |
| 02 | `02_baseline_inference.py` | Captures base model behavior |
| 03 | `03_prepare_subset.py` | Creates a small dataset subset |

These are useful for safety and debugging.

---

## 5. Phase 2B — LoRA & QLoRA Training (Core Pipeline)

These scripts form the **actual fine-tuning pipeline**.

| Order | Script | Purpose |
|------|-------|--------|
| 04 | `04_train_lora.py` | Validate training logic |
| 05 | `05_infer_with_lora.py` | Verify LoRA learning |
| 06 | `06_qlo_ra_preflight.py` | Check 4-bit + GPU readiness |
| 07 | `07_train_qlora.py` | Perform QLoRA training |
| 08 | `08_merge_qlora.py` | Merge adapters into base model |

### Output of Phase 2B

```

outputs/merged_model/

```

This is a standalone Hugging Face model.

---

## 6. Phase 3 — GGUF Conversion & Quantization

This phase prepares the model for **edge / Android deployment**.

### Steps

1. Convert merged Hugging Face model to GGUF (FP16)
2. Quantize GGUF to INT4 (Q4_K_M)

### Final Artifact

```

tinyllama-qlora-q4_k_m.gguf

```

Characteristics:
- INT4
- ~500–550 MB
- CPU-only inference
- Fully offline

---

## 7. Phase 4 — Android / Edge Deployment

- Runtime: `llama.cpp`
- No Python or PyTorch required
- CPU-only inference
- Suitable for Android apps via NDK / JNI

---

## 8. Important Rules

- Do NOT commit model files
- `outputs/` is ignored by git
- RAG logic must not be added to this repository
- Repositories are intentionally separated
- The pipeline is linear and reproducible

---

## 9. Minimal Execution Checklist

- [ ] Clone both repositories
- [ ] Run RAG distillation
- [ ] Copy `final_dataset_instruct.jsonl`
- [ ] (Optional) validate dataset
- [ ] Train with LoRA / QLoRA
- [ ] Merge adapters
- [ ] Convert to GGUF
- [ ] Quantize
- [ ] Deploy on Android

---

## End of README_2
```


