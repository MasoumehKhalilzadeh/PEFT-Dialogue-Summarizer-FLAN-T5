
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

### 📦 Training Details

| Setting             | Value             |
|---------------------|-------------------|
| Model               | FLAN-T5-Base      |
| Batch Size          | 8                 |
| Epochs              | 3                 |
| Samples Used        | 1,000             |
| Eval Samples        | 300               |
| GPU                 | A100 (Colab Pro)  |
| PEFT Method         | LoRA              |
| Train Time          | ~15 minutes       |

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
```


---
### 📉 Training & Validation Loss

| Epoch | Training Loss | Validation Loss |
|-------|----------------|-----------------|
| 1     | 26.49          | 24.95           |
| 2     | 13.24          | 9.63            |
| 3     | 5.88           | 4.66            |



✅ I cut My loss from ~26 to ~5 that’s a very healthy learning curve!
🔽 My loss dropped steadily → My model learned effectively
✅ Validation loss also dropped → it's not overfitting




Model summaries were evaluated using ROUGE on 100 test samples:

### 📊 ROUGE Evaluation

| Metric       | Score   |
|--------------|---------|
| ROUGE-1      | 0.3630  |
| ROUGE-2      | 0.1508  |
| ROUGE-L      | 0.3102  |
| ROUGE-Lsum   | 0.3097  |


The ROUGE scores show that the model did a great job capturing the main ideas and structure of the original summaries. A ROUGE-1 score of 0.3630 means there was a strong match in the words used, while the ROUGE-2 score of 0.1508 shows the model also picked up on meaningful word pairs. The ROUGE-L and ROUGE-Lsum scores, both around 0.31, suggest the model kept the overall flow and structure of the summaries pretty well. Considering this was trained on just 1,000 examples using LoRA — a lightweight fine-tuning method — the results are impressive and show how effective parameter-efficient training can be.



