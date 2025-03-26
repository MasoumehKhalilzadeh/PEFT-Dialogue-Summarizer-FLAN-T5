
# 🧠 Dialogue Summarization with FLAN-T5 + PEFT (LoRA)

This project demonstrates how to efficiently fine-tune a large language model (LLM), `FLAN-T5-base`, using **Parameter-Efficient Fine-Tuning (PEFT)** with **LoRA adapters**, for the task of dialogue summarization. We use the **SAMSum dataset**, which contains multi-turn dialogues and human-written summaries. 

With just **1,000 training samples** the model achieves **strong ROUGE scores**, showing that PEFT can provide high-quality performance with minimal compute and data.

---

## 🔍 Project Highlights

- ✅ Fine-tuned `google/flan-t5-base` using **LoRA** (via Hugging Face PEFT)
- ✅ Trained on **1,000 samples** from the SAMSum dataset
- ✅ Evaluated on 300 test samples with **100 used for ROUGE scoring**
- ✅ Achieved **ROUGE-L: 0.31**, **ROUGE-1: 0.36** — strong results!
- ✅ Runs in under **20 minutes** on Colab Pro (A100 GPU)

---

## 📦 Tech Stack

| Tool          | Purpose                             |
|---------------|-------------------------------------|
| 🤗 Transformers | Load and fine-tune FLAN-T5 model     |
| 🤗 Datasets     | Access and preprocess SAMSum dataset |
| 🤗 PEFT         | Apply LoRA adapters for efficient tuning |
| 🤗 Evaluate     | Compute ROUGE metrics               |
| 🐍 Python       | Core scripting language              |
| 🧠 Google Colab | Environment to train LLMs  |

---

## 📚 Dataset: SAMSum

- **Name**: [SAMSum](https://huggingface.co/datasets/samsum)
- **Samples**: ~15,000 human chat dialogues + summaries
- **Train used**: First 1,000 samples
- **Test used**: First 300 samples
- **Task**: Summarize informal dialogues in English

---

## ⚙️ LoRA Configuration

```python
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)




