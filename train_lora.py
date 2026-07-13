#!/usr/bin/env python3
"""Newsio+ QLoRA 訓練腳本 — 微調 Llama 3.2 3B 做英文 → 繁體中文（zh-TW）新聞翻譯。

流程：
  1. 讀取 data/train.jsonl（欄位：en_title / en_bullets / zh_title / zh_bullets ...）
  2. 用 Llama 3.2 的 chat template 組成「system + user(英文) → assistant(繁中)」配對
  3. 以 4-bit（nf4）載入基底模型，掛上 LoRA adapter（只動 q/k/v/o 四個 projection）
  4. 訓練，把 adapter 存到 out/adapter/

用法：
  python train_lora.py                       # 預設 r=16、1 epoch、全部資料
  python train_lora.py --rank 8 --epochs 2 --max-samples 2000

預設會：
  * 丟掉「沒有繁中重點（zh_bullets 為空）」的資料（--keep-empty-bullets 可保留）
  * 切 5% 出來當驗證集，每個 epoch 印一次 eval loss（--val-frac 0 可關掉）
"""

import argparse
import json
import os
from dataclasses import dataclass

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

# ---------------------------------------------------------------------------
# 常數：這些必須和 Newsio+ 網站教的內容一致，不要隨意更改。
# ---------------------------------------------------------------------------

# 固定的 system prompt — 訓練與推論都要用同一句，模型才知道「現在是翻譯任務」。
SYSTEM_PROMPT = (
    "你是 Newsio 的新聞翻譯編輯。"
    "將英文新聞標題與重點翻譯成台灣慣用的繁體中文，保持 Newsio 簡潔風格。"
)

# LoRA 只掛在 attention 的四個 projection 上（課程教的就是這四個）。
# 對 Llama 3.2 3B（hidden=3072、GQA 8 組 KV head、head_dim=128）而言，每個 block：
#   q_proj: r*(3072+3072)、k_proj: r*(3072+1024)、v_proj: r*(3072+1024)、o_proj: r*(3072+3072)
#   → 合計 r × 20,480 個參數；全模型 28 個 block → r × 573,440。
#   r=16 時就是 9,175,040 個可訓練參數（約佔 3.21B 的 0.29%）。
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

DEFAULT_MODEL = "unsloth/Llama-3.2-3B-Instruct"  # 不用申請權限的鏡像
DATA_PATH = os.path.join("data", "train.jsonl")
ADAPTER_DIR = os.path.join("out", "adapter")
MAX_LENGTH = 1024  # 單筆樣本的 token 上限（Newsio 摘要都很短，1024 綽綽有餘）


# ---------------------------------------------------------------------------
# 資料處理
# ---------------------------------------------------------------------------

def as_lines(value):
    """en_bullets / zh_bullets 可能是 list 也可能是字串 — 一律轉成「- 重點」逐行文字。"""
    if isinstance(value, str):
        value = [value]
    return "\n".join(f"- {item}".strip() for item in value if str(item).strip())


def build_messages(record):
    """把一筆 train.jsonl 資料組成 chat 訊息（user=英文原文、assistant=繁中翻譯）。"""
    user_text = record["en_title"].strip() + "\n" + as_lines(record["en_bullets"])
    assistant_text = record["zh_title"].strip() + "\n" + as_lines(record["zh_bullets"])
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]


def load_records(path, max_samples=None):
    if not os.path.exists(path):
        raise SystemExit(
            f"找不到 {path}。\n"
            "請先到 https://plus.newsio.io/data.html 用 Google 登入，"
            "train.jsonl 會寄到你的信箱，下載後放進 data/ 資料夾。"
        )
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
            if max_samples is not None and len(records) >= max_samples:
                break
    return records


def tokenize_record(record, tokenizer):
    """Tokenize 一筆資料，並把 prompt（system+user）部分的 label 遮成 -100。

    模型只需要學 assistant 的輸出（翻譯本身），所以 prompt 部分不計算 loss —
    這就是所謂的 completion-only 訓練。
    """
    messages = build_messages(record)

    # 完整對話（含 assistant 答案）— 訓練用，不加 generation prompt。
    # 注意：transformers 5.x 下 tokenize=True 會回傳 BatchEncoding（dict），
    # 要取 ['input_ids'] 才是 list[int]（舊版直接回傳 list[int]）。
    full_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, return_dict=True
    )["input_ids"]
    # 只有 prompt（system+user）+ assistant 開頭標記 — 用來知道要遮住幾個 token。
    prompt_ids = tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True, return_dict=True
    )["input_ids"]

    full_ids = full_ids[:MAX_LENGTH]
    labels = list(full_ids)
    prompt_len = min(len(prompt_ids), len(full_ids))
    labels[:prompt_len] = [-100] * prompt_len  # -100 = 這些位置不算 loss

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "prompt_len": prompt_len,
    }


@dataclass
class PadCollator:
    """把一個 batch 內長短不一的樣本補到等長（labels 用 -100 補，不影響 loss）。"""

    pad_token_id: int

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad = max_len - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [self.pad_token_id] * pad)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad)
            batch["labels"].append(f["labels"] + [-100] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


# ---------------------------------------------------------------------------
# 主程式
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Newsio+ QLoRA 訓練腳本")
    parser.add_argument("--rank", type=int, default=16, choices=range(1, 65),
                        metavar="[1-64]", help="LoRA rank（r），預設 16；alpha 固定為 2×r")
    parser.add_argument("--epochs", type=float, default=1.0, help="訓練幾個 epoch，預設 1")
    parser.add_argument("--batch", type=int, default=2, help="每張 GPU 的 batch size，預設 2")
    parser.add_argument("--lr", type=float, default=2e-4, help="學習率，預設 2e-4")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="只用前 N 筆資料（預設全部 14,746 筆）")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"基底模型，預設 {DEFAULT_MODEL}")
    parser.add_argument("--data", type=str, default=DATA_PATH,
                        help=f"訓練資料路徑，預設 {DATA_PATH}")
    parser.add_argument("--out", type=str, default=ADAPTER_DIR,
                        help=f"adapter 輸出資料夾，預設 {ADAPTER_DIR}")
    parser.add_argument("--val-frac", type=float, default=0.05,
                        help="切多少比例當驗證集（每 epoch 印 eval loss）；預設 0.05，設 0 關掉")
    parser.add_argument("--keep-empty-bullets", action="store_true",
                        help="保留 zh_bullets 為空的資料（預設會丟掉，避免模型學會不輸出重點）")
    return parser.parse_args()


def clean_records(records):
    """丟掉沒有繁中重點的資料 — 約 12%% 的原始資料 zh_bullets 是空的，
    留著會教模型「有時候不用輸出重點」。回傳 (乾淨資料, 丟掉筆數)。"""
    kept = [r for r in records
            if (r.get("zh_bullets") or []) and str(r.get("zh_title") or "").strip()]
    return kept, len(records) - len(kept)


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "找不到 CUDA GPU。4-bit QLoRA 訓練需要 NVIDIA GPU；"
            "沒有的話請改用免費的 Colab T4（見 README 的 Colab 章節）。"
        )
    use_bf16 = torch.cuda.is_bf16_supported()  # T4 只支援 fp16，較新的卡用 bf16

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Llama 沒有 pad token，借用 eos

    # ---- 資料 ----
    print(f"載入 {args.data} ...")
    records = load_records(args.data, args.max_samples)
    if not args.keep_empty_bullets:
        records, dropped = clean_records(records)
        print(f"資料清理：丟掉 {dropped:,} 筆空 zh_bullets 的資料"
              f"（加 --keep-empty-bullets 可保留）")
    print(f"共 {len(records):,} 筆訓練配對")

    dataset = Dataset.from_list(records)
    dataset = dataset.map(
        lambda rec: tokenize_record(rec, tokenizer),
        remove_columns=dataset.column_names,
        desc="Tokenize + 遮罩 prompt",
    )
    # 保險起見：prompt 就把長度吃滿的樣本（assistant 完全被截掉）直接丟掉。
    dataset = dataset.filter(lambda ex: ex["prompt_len"] < len(ex["input_ids"]))
    dataset = dataset.remove_columns(["prompt_len"])

    # ---- 切訓練 / 驗證集 ----
    # 拿一小塊「訓練時不會看到」的資料當驗證集，才能判斷有沒有 overfit。
    eval_dataset = None
    if args.val_frac and args.val_frac > 0 and len(dataset) >= 20:
        split = dataset.train_test_split(test_size=args.val_frac, seed=42)
        dataset, eval_dataset = split["train"], split["test"]
        print(f"切出驗證集：訓練 {len(dataset):,} 筆 / 驗證 {len(eval_dataset):,} 筆")

    # ---- 基底模型（4-bit nf4 量化 → 3B 模型約 2GB VRAM）----
    print(f"以 4-bit 載入 {args.model} ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False  # 訓練時關掉 KV cache（和 gradient checkpointing 衝突）
    model = prepare_model_for_kbit_training(model)

    # ---- LoRA adapter ----
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,       # 課程慣例：alpha = 2×r
        target_modules=TARGET_MODULES,  # 只動 q/k/v/o
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # get_nb_trainable_parameters() 回傳 (可訓練參數, 含 adapter 的總參數)；
    # 減掉 trainable 才是「基底模型」的參數量（Llama 3.2 3B = 3,212,749,824）。
    trainable, total_with_adapter = model.get_nb_trainable_parameters()
    base_params = total_with_adapter - trainable
    print(
        f"\nLoRA r={args.rank}: 可訓練參數 {trainable:,}，"
        f"基底模型 {base_params:,}（佔 {100 * trainable / base_params:.2f}%）"
    )
    print(f"換算：每個 block r×20,480 = {args.rank * 20480:,}，×28 blocks = {args.rank * 20480 * 28:,}\n")

    # ---- 訓練 ----
    # 有效 batch size 固定為 16：VRAM 小就把 --batch 調低，
    # gradient accumulation 會自動補上（例如 batch=1 → 累積 16 步才更新一次）。
    grad_accum = max(1, 16 // args.batch)
    training_args = TrainingArguments(
        output_dir=os.path.join(args.out, "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=20,
        # 有驗證集就每個 epoch 評估一次，印出 eval loss（看有沒有 overfit）。
        eval_strategy="epoch" if eval_dataset is not None else "no",
        bf16=use_bf16,
        fp16=not use_bf16,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="no",   # 只在最後存 adapter，不存中間 checkpoint
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=PadCollator(pad_token_id=tokenizer.pad_token_id),
    )
    trainer.train()

    # ---- 存檔 ----
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"\n完成！adapter 已存到 {args.out}/")
    print("下一步：python test_translate.py 試翻一則新聞。")


if __name__ == "__main__":
    main()
