# PEFT Dialogue Summarizer FLAN-T5
A project using PEFT (LoRA) to fine-tune FLAN-T5 for multi-turn dialogue summarization on the SAMSum dataset.


# 🧠 Dialogue Summarizer with FLAN-T5 + PEFT (LoRA)

This project shows how to use **Parameter Efficient Fine-Tuning (PEFT)** with **LoRA adapters** to fine-tune the `FLAN-T5` model for **multi-turn dialogue summarization** using the [SAMSum dataset](https://huggingface.co/datasets/samsum).


---

## 🧩 What’s Inside

- 🤖 Model: `google/flan-t5-base`
- 📚 Dataset: `samsum` (chat dialogue + summaries)
- ⚡ Fine-Tuning Method: **PEFT (LoRA)**
- 🛠️ Framework: Hugging Face Transformers
- 🚀 Runtime: Google Colab 


---

## 🔍 Sample Results

**🗨️ Input Dialogue:**

Mom: Are you coming for dinner?
Me: I can't, I'm still at work.
Mom: OK, I’ll save you some food.
Me: Thanks, love you!


**✅ Real Summary:**
> Mom asks about dinner; child replies they're working late.

**🤖 Model Summary (PEFT):**
> Person is working late and Mom will save dinner for them.

---


## 📈 What I Learned

- How to fine-tune large language models **efficiently**
- How **LoRA** reduces training cost while keeping good performance
- How to use Hugging Face’s `transformers`, `datasets`, and `peft` libraries together

---

## 📚 Dataset Info

- **SAMSum Dataset**  
  > 16,000+ short chat conversations with human-written summaries  
  📦 [View on Hugging Face](https://huggingface.co/datasets/samsum)

---

## 🔧 Requirements

Install needed libraries:
```bash
pip install transformers datasets peft accelerate evaluate sentencepiece py7zr

