"""LoRA fine-tune of a small instruct model with Unsloth. Run on a free Colab or Kaggle GPU.

Colab setup (first cell):
    !pip install unsloth trl datasets
    # upload finetune/data/train.jsonl

Then:
    !python train_lora.py --data train.jsonl --base unsloth/Qwen2.5-1.5B-Instruct

Check the Unsloth model list for newer small instruct models before you start; any
1-3B instruct model with a chat template works. Library APIs change often: if a call
fails, compare it with the current Unsloth example notebooks.

After training, copy the merged folder to finetune/outputs/marketpulse-lora-merged,
set generation.backend: hf in configs/config.yaml, and run
    python -m marketpulse.cli evaluate --backend hf
on dates after your split date only.
"""

from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="train.jsonl")
    p.add_argument("--base", default="unsloth/Qwen2.5-1.5B-Instruct")
    p.add_argument("--out", default="marketpulse-lora")
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--max-seq-len", type=int, default=2048)
    args = p.parse_args()

    from unsloth import FastLanguageModel  # import first: it patches transformers
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base, max_seq_length=args.max_seq_len, load_in_4bit=True)
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=42)

    ds = load_dataset("json", data_files=args.data, split="train")
    ds = ds.map(lambda ex: {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False)})
    split = ds.train_test_split(test_size=0.1, seed=42)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        args=SFTConfig(
            dataset_text_field="text", max_seq_length=args.max_seq_len,
            per_device_train_batch_size=2, gradient_accumulation_steps=4,
            num_train_epochs=args.epochs, learning_rate=2e-4, warmup_ratio=0.05,
            lr_scheduler_type="cosine", logging_steps=10, eval_strategy="epoch",
            output_dir=args.out, seed=42, report_to="none"),
    )
    trainer.train()
    model.save_pretrained_merged(args.out + "-merged", tokenizer, save_method="merged_16bit")
    print(f"Saved merged model to {args.out}-merged")


if __name__ == "__main__":
    main()
